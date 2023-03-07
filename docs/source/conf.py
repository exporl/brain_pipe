# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

root_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Brain Pipe"
copyright = "2023, Bernd Accou, ExpORL"
author = "Bernd Accou, ExpORL"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    # 'autoapi.extension'
]


templates_path = ["_templates"]
exclude_patterns = []

add_module_names = False

# autoapi_type = 'python'
# autoapi_dirs = [os.path.join(root_folder, 'brain_pipe')]
# autoapi_template_dir = '_autoapi_templates'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_material"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Set link name generated in the top bar.
html_title = "Brain Pipe"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "Brain Pipe",
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # 'base_url': 'https://project.github.io/project',
    # Set the color and the accent color
    "color_primary": "blue",
    "color_accent": "light-blue",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/exporl/brain_pipe",
    "repo_name": "Brain Pipe",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 2,
    # If False, expand all TOC entries
    "globaltoc_collapse": False,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
    "touch_icon": "_images/brain_pipe.svg",
}
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}


intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}
numfig = True
autosummary_generate = True
html_logo = "_images/brain_pipe.svg"
