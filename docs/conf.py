"""Build the handbook from Markdown, public source APIs and schema tables."""
from pathlib import Path
import ast
import os
import sys

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS / "_ext"))

project = "hwoslaps"
author = "Georgios Vassilakis"
copyright = "2025, California Institute of Technology"
version_tree = ast.parse((ROOT / "src/hwoslaps/_version.py").read_text())
release = next(ast.literal_eval(node.value) for node in version_tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "__version__"
                       for target in node.targets))
version = release

extensions = [
    "myst_parser", "sphinx.ext.autodoc", "autoapi.extension", "sphinx.ext.napoleon",
    "sphinx_copybutton", "generate_reference",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "_ext", "_generated", "README.md", "CONFIG.md"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
napoleon_numpy_docstring = True
napoleon_google_docstring = True

# AutoAPI parses source files; it does not import the optional science stack.
# The registry extension selects the public symbols declared by package owners.
autoapi_dirs = [str(ROOT / "src/hwoslaps")]
autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
autoapi_options = ["members", "undoc-members", "show-inheritance"]
autoapi_keep_files = False
autodoc_typehints = "signature"
autodoc_member_order = "bysource"

html_theme = "furo"
html_title = "hwoslaps"
html_static_path = ["_static"]
html_css_files = ["handbook.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#246c7a",
        "color-brand-content": "#246c7a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#86cbd5",
        "color-brand-content": "#86cbd5",
    },
    "navigation_with_keys": True,
}
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_show_sourcelink = True
html_copy_source = True
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
