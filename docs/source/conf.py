# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NAVTILVS'
copyright = '2026, Filippo Di Ludovico'
author = 'Filippo Di Ludovico'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

autodoc_member_order = 'bysource'

# Napoleon settings to format Attributes cleanly as bulleted field lists
napoleon_use_ivar = True
napoleon_use_param = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'page_width': '960px',
    'sidebar_width': '230px',
    'font_family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'head_font_family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'code_font_family': 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    'show_relbars': True,
}

html_css_files = [
    'custom.css',
]
