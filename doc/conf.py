# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

# -- Project information -----------------------------------------------------

project = 'wilfried library'
copyright = '2021, Wilfried Mercier'
author = 'Wilfried Mercier'
show_authors=True

highlight_options = {
  'default': {'lexers.python.PythonLexer'},
}

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.imgmath', 'sphinx.ext.viewcode']

# The full version, including alpha/beta/rc tags
release = '1.0'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'

html_theme_options = {"repository_url": "https://github.com/WilfriedMercier/wilfried",
                      "use_repository_button": True,
                      "use_edit_page_button": True,
                      "home_page_in_toc": True
                     }

html_sidebars = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

