"""Installer preflight never replaces prefixes or guesses after lookup failure.

Conda and the selected environment interpreter are isolated executable doubles;
these tests never create real conda environments or install scientific packages.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def installer(tmp_path):
    binary = tmp_path / "bin"
    base = tmp_path / "base"
    envs = tmp_path / "envs"
    binary.mkdir()
    envs.mkdir()
    (base / "bin").mkdir(parents=True)
    (base / "etc/profile.d").mkdir(parents=True)
    # No unversioned Python is exposed on PATH. JSON is parsed with base/bin/python.
    (binary / "dirname").symlink_to(shutil.which("dirname"))
    (base / "bin/python").symlink_to(sys.executable)
    log = tmp_path / "commands.log"
    stub = tmp_path / "environment-python"
    stub.write_text('#!/bin/bash\nprintf "interpreter %s\\n" "$*" >> "$TASK_TEST_LOG"\n')
    stub.chmod(0o755)
    conda = binary / "conda"
    conda.write_text('''#!/bin/bash
printf 'conda %s\n' "$*" >> "$TASK_TEST_LOG"
case "$*" in
    'info --base') printf '%s\n' "$TASK_TEST_BASE";;
    'info --json') printf '%s\n' "$TASK_TEST_INFO";;
    'env list --json')
        if [ "$TASK_TEST_LOOKUP_FAILURE" = 1 ]; then exit 23; fi
        printf '%s\n' "$TASK_TEST_ENVS";;
    create*)
        [ "$2" = --prefix ] || exit 41
        # Fail even the double if the installer tries to replace an existing path.
        [ ! -e "$3" ] && [ ! -L "$3" ] || exit 42
        /bin/mkdir -p "$3/conda-meta" "$3/bin"
        /bin/cp "$TASK_TEST_STUB" "$3/bin/python";;
    *) exit 43;;
esac
''')
    conda.chmod(0o755)
    (base / "etc/profile.d/conda.sh").write_text('''conda() {
    if [ "$1" = activate ]; then
        printf 'activate %s\n' "$2" >> "$TASK_TEST_LOG"
        export CONDA_PREFIX="$2"
    else
        command conda "$@"
    fi
}
''')
    environment = {
        **os.environ,
        "PATH": str(binary),
        "TASK_TEST_BASE": str(base),
        "TASK_TEST_INFO": json.dumps({"root_prefix": str(base), "envs_dirs": [str(envs)]}),
        "TASK_TEST_ENVS": json.dumps({"envs": []}),
        "TASK_TEST_LOOKUP_FAILURE": "0",
        "TASK_TEST_STUB": str(stub),
        "TASK_TEST_LOG": str(log),
    }

    def run(*arguments, **changes):
        result = subprocess.run(["/bin/bash", str(ROOT / "install.sh"), *arguments],
                                env={**environment, **changes}, text=True, capture_output=True, timeout=20)
        commands = log.read_text().splitlines() if log.exists() else []
        return result, commands

    return run, envs, base, stub


@pytest.mark.parametrize("lookup", ["failure", "invalid_json", "invalid_inventory"])
def test_failed_environment_inventory_never_creates_or_installs(installer, lookup):
    run, envs, _, _ = installer
    changes = {"failure": {"TASK_TEST_LOOKUP_FAILURE": "1"},
               "invalid_json": {"TASK_TEST_ENVS": "not JSON"},
               "invalid_inventory": {"TASK_TEST_ENVS": '{"envs": null}'}}[lookup]
    result, commands = run(**changes)
    assert result.returncode != 0
    assert not any(line.startswith(("conda create", "activate", "interpreter")) for line in commands)
    assert not (envs / "hwo-slaps").exists()


@pytest.mark.parametrize("kind", ["registered", "unregistered", "partial", "file", "symlink"])
@pytest.mark.parametrize("selection", ["name", "prefix"])
def test_existing_prefix_is_preserved_without_reuse(installer, kind, selection):
    run, envs, _, stub = installer
    prefix = envs / "hwo-slaps"
    if kind == "file":
        prefix.write_text("existing user file")
    elif kind == "symlink":
        prefix.symlink_to(envs / "absent")
    else:
        prefix.mkdir()
        (prefix / "keep.txt").write_text("existing environment data")
        if kind in {"registered", "unregistered"}:
            (prefix / "conda-meta").mkdir()
            (prefix / "bin").mkdir()
            shutil.copy(stub, prefix / "bin/python")
    changes = {"TASK_TEST_ENVS": json.dumps({"envs": [str(prefix)]})} if kind == "registered" else {}
    arguments = ["--prefix", str(prefix)] if selection == "prefix" else []
    before = prefix.read_bytes() if kind == "file" else None
    result, commands = run(*arguments, **changes)
    assert result.returncode != 0
    assert "Existing prefix preserved" in result.stderr
    assert not any(line.startswith(("conda create", "activate", "interpreter")) for line in commands)
    if kind == "file":
        assert prefix.read_bytes() == before
    elif kind == "symlink":
        assert prefix.is_symlink()
    else:
        assert (prefix / "keep.txt").read_text() == "existing environment data"


def test_absent_named_environment_uses_base_python_and_exact_prefix(installer):
    run, envs, _, _ = installer
    result, commands = run()
    assert result.returncode == 0, result.stderr
    prefix = envs / "hwo-slaps"
    assert f"conda create --prefix {prefix} python=3.11 -y" in commands
    assert f"activate {prefix}" in commands
    assert any(line.startswith("interpreter -m pip install -c ") for line in commands)
    assert (prefix / "conda-meta").is_dir()


def test_reuse_requires_a_real_existing_prefix_and_never_recreates(installer):
    run, envs, _, stub = installer
    prefix = envs / "hwo-slaps"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    shutil.copy(stub, prefix / "bin/python")
    result, commands = run("--reuse")
    assert result.returncode == 0, result.stderr
    assert not any(line.startswith("conda create") for line in commands)
    assert f"activate {prefix}" in commands
    assert any(line.startswith("interpreter -m pip install -c ") for line in commands)


def test_reuse_of_absent_prefix_fails_before_creation(installer):
    run, envs, _, _ = installer
    result, commands = run("--reuse")
    assert result.returncode != 0
    assert "--reuse requires an existing environment" in result.stderr
    assert not any(line.startswith(("conda create", "interpreter")) for line in commands)
    assert not (envs / "hwo-slaps").exists()


@pytest.mark.parametrize("version", ["3.10", "3.13", "3.14", "3"])
def test_unsupported_scientific_python_fails_before_conda(installer, version):
    run, _, _, _ = installer
    result, commands = run("--python", version)
    assert result.returncode == 2
    assert "requires Python 3.11 or 3.12" in result.stderr
    assert not commands


def test_base_prefix_is_never_modified_even_with_reuse(installer):
    run, _, base, _ = installer
    result, commands = run("--prefix", str(base), "--reuse")
    assert result.returncode != 0
    assert "base environment" in result.stderr
    assert not any(line.startswith(("conda create", "activate", "interpreter")) for line in commands)


def test_reuse_rejects_an_interpreter_from_another_prefix_before_pip(installer):
    run, envs, _, _ = installer
    prefix = envs / "hwo-slaps"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "bin/python").symlink_to(sys.executable)
    result, commands = run("--reuse")
    assert result.returncode != 0
    assert "does not belong to the selected environment" in result.stderr
    assert not any("pip install" in line or line.startswith("conda create") for line in commands)
