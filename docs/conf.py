# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

# Add the parent directory to the path so we can import tauclean
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project information
project = "tauclean"
copyright = "2024, Bradley Meyers"
author = "Bradley Meyers, Ramesh Bhat, Olivia Young, Michael Lam"
release = "1.0.0"

# Extensions to use
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that should be excluded.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use
html_theme = "sphinx_rtd_theme"

# Theme options
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#2980b3",
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "taucleandoc"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
}

# Autosummary settings
autosummary_generate = True

# Napoleon settings (for better docstring formatting)
napoleon_google_docstring = False
napoleon_numpy_docstring = False
napoleon_use_rtype = True

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Mathjax settings
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Source file suffix
source_suffix = ".rst"

# Master doc
master_doc = "index"
