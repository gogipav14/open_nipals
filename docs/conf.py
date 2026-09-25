# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src/open_nipals/"))

project = "open_nipals"
copyright = "2025, David R. Ochsenbein, Niels Schlusser, Ryan M. Wall"
author = "David R. Ochsenbein, Niels Schlusser, Ryan M. Wall"
release = "2.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# enable automated creation of docs with autodoc
# conversion of numpy/autodocstring style to
# rst readable style with napoleon
# rendering markdown with myst
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser"]

# The JAX module is an optional extra; document it without installing JAX
autodoc_mock_imports = ["jax", "jaxlib"]

# NipalsPCA and NipalsPLS exist both in the NumPy and in the JAX module,
# so short names in type hints match two classes; either link is fine
suppress_warnings = ["ref.python"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
