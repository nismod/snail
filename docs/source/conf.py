# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information --
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "snail"
copyright = "2023, snail contributors"
author = "snail contributors"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""

try:
    from snail import __version__

    version = __version__
except ImportError:
    pass
else:
    release = version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extra styles, found in _static
def setup(app):
    app.add_css_file("theme_tweaks.css")


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "m2r2",
]
templates_path = ["_templates"]
exclude_patterns = []

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"

html_title = project
html_short_title = project
html_static_path = ["_static"]
html_theme_options = {"show_powered_by": False}
