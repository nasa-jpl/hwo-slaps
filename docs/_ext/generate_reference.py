"""Generate curated AutoAPI pages without importing the scientific package.

AutoAPI must parse the source first (builder-inited priority 500). This extension
uses priority 900, validates every public target against its static inventory, then
writes pages before Sphinx discovers documents. Configuration uses the real CLI in
a separate process and must equal the tracked CONFIG.md byte for byte.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys


def _assignments(tree):
    result = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name) and node.value is not None:
                result[target.id] = node.value
    return result


def _source_registry(package):
    """Return public alias -> source definition, preserving no executable values."""
    modules = {}
    for path in sorted(package.rglob('*.py')):
        parts = list(path.relative_to(package.parent).with_suffix('').parts)
        is_package = parts[-1] == '__init__'
        if is_package:
            parts.pop()
        name = '.'.join(parts)
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        assigned = _assignments(tree)
        definitions = set(assigned)
        definitions.update(node.name for node in tree.body
                           if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
        imports = {}
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            parent = parts if is_package else parts[:-1]
            if node.level:
                parent = parent[:len(parent) - node.level + 1]
                owner = '.'.join(parent + (node.module.split('.') if node.module else []))
            else:
                owner = node.module or ''
            for item in node.names:
                if item.name != '*':
                    imports[item.asname or item.name] = owner + '.' + item.name
        modules[name] = (assigned, definitions, imports, path)

    aliases = {}
    for module, (assigned, definitions, imports, path) in modules.items():
        registry = assigned.get('_PUBLIC_API', assigned.get('_EXPORTS'))
        if registry is not None:
            entries = ast.literal_eval(registry)
            if not isinstance(entries, dict):
                raise ValueError(f'{path}: public registry must be a literal mapping')
            for public, target in entries.items():
                if not isinstance(public, str):
                    raise ValueError(f'{path}: public name must be text')
                if isinstance(target, tuple) and len(target) == 2 and all(isinstance(x, str) for x in target):
                    child, member = target
                    canonical = module + '.' + child + '.' + member
                elif isinstance(target, str) and target.startswith('.'):
                    canonical = module + target + '.' + public
                else:
                    raise ValueError(f'{path}: unsupported public registry entry {public!r}')
                aliases[module + '.' + public] = canonical
        public_all = assigned.get('__all__')
        if public_all is not None:
            try:
                names = ast.literal_eval(public_all)
            except (ValueError, TypeError):
                if registry is None:
                    raise ValueError(f'{path}: nonliteral __all__ has no supported registry') from None
                # Lazy package expressions are owned by the literal registry.
                # Explicit literal extras (currently __version__) remain public.
                names = [item.value for item in getattr(public_all, 'elts', ())
                         if isinstance(item, ast.Constant) and isinstance(item.value, str)]
            if not isinstance(names, (list, tuple)) or not all(isinstance(x, str) for x in names):
                raise ValueError(f'{path}: __all__ must name strings')
            for name in names:
                aliases.setdefault(module + '.' + name, module + '.' + name)

    def resolve(target, seen):
        if target in seen:
            raise ValueError(f'public alias cycle: {target}')
        module, _, member = target.rpartition('.')
        if module not in modules:
            raise ValueError(f'public target is outside parsed package: {target}')
        _, definitions, imports, path = modules[module]
        if member in definitions:
            return target
        if member in imports:
            return resolve(imports[member], seen | {target})
        if target in aliases and aliases[target] != target:
            return resolve(aliases[target], seen | {target})
        raise ValueError(f'{path}: missing public source definition {member}')

    return {alias: resolve(target, set()) for alias, target in aliases.items()}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8')


def _private(target):
    return any(part.startswith('_') for part in target.split('.'))


def _api_pages(app, package):
    aliases = _source_registry(package)
    objects = getattr(app.env, 'autoapi_all_objects', None)
    if not objects:
        raise RuntimeError('AutoAPI inventory is absent; load autoapi.extension before generate_reference')
    missing = sorted(set(aliases.values()) - objects.keys())
    if missing:
        raise RuntimeError('public targets missing from AutoAPI inventory: ' + ', '.join(missing))
    aliases = {alias: target for alias, target in aliases.items() if not _private(target)}
    groups = defaultdict(list)
    for target in sorted(set(aliases.values())):
        groups[target.rpartition('.')[0]].append(target)
    output = Path(app.srcdir) / 'api'
    page_names = {module + '.rst' for module in groups} | {'index.rst'}
    if output.exists():
        for path in output.glob('*.rst'):
            if path.name not in page_names:
                if not path.read_text(encoding='utf-8').startswith('.. Generated by generate_reference;'):
                    raise RuntimeError(f'refusing to remove an authored API page: {path}')
                path.unlink()
    index = ['.. Generated by generate_reference; do not edit.', '',
             'All modules', '===========', '',
             'Every public function and class, by the module that defines it.', '',
             '.. toctree::', '   :maxdepth: 1', '']
    index += ['   ' + module for module in groups]
    index += ['', 'Shorter import paths', '--------------------', '',
              'These names can also be imported from a package, for example',
              '``from hwoslaps import forecast``.', '']
    for alias, target in sorted(aliases.items()):
        if alias != target:
            index.append(f'* ``{alias}`` → :py:obj:`{target}`')
    _write(output / 'index.rst', '\n'.join(index) + '\n')
    directive_types = {'function': 'function', 'class': 'class', 'exception': 'exception',
                       'data': 'data', 'attribute': 'data'}
    for module, targets in groups.items():
        lines = ['.. Generated by generate_reference; do not edit.', '',
                 module, '=' * len(module), '']
        for target in targets:
            kind = getattr(objects[target], 'type', None)
            if kind not in directive_types:
                raise RuntimeError(f'unsupported public AutoAPI object type: {target}: {kind}')
            lines += [f'.. autoapi{directive_types[kind]}:: {target}']
            if kind in ('class', 'exception'):
                lines += ['   :members:']
            lines += ['']
        _write(output / (module + '.rst'), '\n'.join(lines))


def _without_nested_engine_config(text):
    """Drop the batch.config sections, which repeat the engine configuration verbatim.

    Headings such as ``scene.lens.mass.<name>`` escape ``<`` so Markdown keeps the placeholder.
    """
    sections = text.split('\n## ')
    kept = [sections[0]] + [section for section in sections[1:] if not section.startswith('batch.config')]
    page = '\n## '.join(kept)
    return '\n'.join(line.replace('<', '\\<') if line.startswith('## ') else line for line in page.split('\n'))


def _configuration(app, root):
    python = app.config.reference_cli_python or sys.executable
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src')
    environment['PYTHONNOUSERSITE'] = '1'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([python, '-m', 'hwoslaps', 'reference'], cwd=root, env=environment,
                            capture_output=True, check=True, timeout=120)
    canonical = root / 'docs' / 'CONFIG.md'
    if result.stdout != canonical.read_bytes():
        raise RuntimeError('docs/CONFIG.md is out of date; run: python -m hwoslaps reference > docs/CONFIG.md')
    destination = Path(app.srcdir) / '_generated' / 'configuration.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    page = _without_nested_engine_config(result.stdout.decode('utf-8')).encode('utf-8')
    if not destination.exists() or destination.read_bytes() != page:
        destination.write_bytes(page)
    sections = []
    for command in ((), ('validate',), ('forecast',), ('simulate',), ('reference',),
                    ('batch',), ('batch', 'plan'), ('batch', 'run'), ('batch', 'status')):
        help_result = subprocess.run([python, '-m', 'hwoslaps', *command, '--help'],
                                     cwd=root, env=environment, capture_output=True,
                                     check=True, text=True, timeout=30)
        label = ' '.join(('hwoslaps', *command))
        sections += [f'### {label}', '', '```text', help_result.stdout.rstrip(), '```', '']
    _write(destination.with_name('command_line.md'), '\n'.join(sections))


def _generate(app):
    root = Path(app.confdir).resolve().parent
    package = root / 'src' / 'hwoslaps'
    if not package.is_dir():
        raise RuntimeError(f'expected docs/conf.py beside src/hwoslaps: {package}')
    _api_pages(app, package)
    _configuration(app, root)


def _include_generated(app, docname, source):
    fragments = {
        'configuration': ('GENERATED_CONFIGURATION', 'configuration.md'),
        'command_line': ('GENERATED_COMMAND_LINE', 'command_line.md'),
    }
    if docname in fragments:
        token, filename = fragments[docname]
        marker = f'<!-- {token} -->'
        if source[0].count(marker) != 1:
            raise RuntimeError(f'{docname}: expected one generated-reference marker')
        path = Path(app.srcdir) / '_generated' / filename
        source[0] = source[0].replace(marker, path.read_text(encoding='utf-8'))
        app.env.note_dependency(str(path))
    elif docname.startswith('api/') and docname != 'api/index':
        module = docname.removeprefix('api/')
        path = Path(app.confdir).parent / 'src' / (module.replace('.', '/') + '.py')
        if not path.is_file():
            path = path.with_suffix('') / '__init__.py'
        if not path.is_file():
            raise RuntimeError(f'generated API has no source dependency: {module}')
        app.env.note_dependency(str(path))


def setup(app):
    app.add_config_value('reference_cli_python', None, 'env')
    app.connect('builder-inited', _generate, priority=900)
    app.connect('source-read', _include_generated)
    return {'version': '1', 'parallel_read_safe': True, 'parallel_write_safe': True}
