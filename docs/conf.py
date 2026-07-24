# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# flake8: noqa
import os
import sys

# Make the f1tenth_gym package importable for autodoc (repo root is one up).
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "f1tenth_gym"
copyright = "2020-2024, F1TENTH Foundation"
author = "F1TENTH Foundation"
release = "dev-humble"
version = "dev-humble"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",       # [source] links
    "sphinx.ext.intersphinx",    # cross-links to numpy/gymnasium
    "sphinx.ext.autosectionlabel",
]

source_suffix = ".rst"
source_encoding = "utf-8-sig"
master_doc = "index"
language = os.getenv("READTHEDOCS_LANGUAGE", "en")
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Avoid duplicate-label warnings when the same section title appears on several
# pages (autosectionlabel scopes labels by document).
autosectionlabel_prefix_document = True

# -- Autodoc -----------------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# GUI / native-graphics deps that need not (and on a headless RTD builder cannot)
# be imported just to read docstrings. numba/scipy/numpy/gymnasium are real deps
# and are installed for the build.
autodoc_mock_imports = ["pyqtgraph", "PyQt6", "OpenGL"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# Render "Attributes:" docstring sections as inline :ivar: fields rather than
# separate object descriptions, so dataclass fields aren't documented twice
# (once as an autodoc member, once from the Attributes section).
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "collapse_navigation": False,
    "prev_next_buttons_location": "bottom",
    "navigation_depth": 4,
}
html_context = {
    "display_github": True,
    "github_user": "f1tenth",
    "github_repo": "f1tenth_gym",
    "github_version": "dev-humble",
    "conf_py_path": "/docs/",
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["css/custom.js"]
html_logo = "assets/f1tenth_gym.svg"
html_favicon = "assets/f1_stickers_02.png"
