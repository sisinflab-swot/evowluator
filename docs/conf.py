import os
import sys

sys.path.extend([
    os.path.abspath('..'),
    os.path.abspath('../lib/pyutils')
])

# Project metadata

project = 'evOWLuator'
copyright = '2019, SisInf Lab'
author = 'SisInf Lab'
version = '0.1.1'
release = version
logo = ''
git_url = 'https://github.com/sisinflab-swot/evowluator'

# Sphinx

primary_domain = 'py'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints'
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

rst_prolog = """
:github_url: {}
""".format(git_url)

rst_epilog = """
.. |syntaxes| replace:: :code:`functional`, :code:`manchester`, :code:`owlxml`, :code:`rdfxml`
.. _git_url: {}
""".format(git_url)

# Autodoc

autodoc_default_options = {
    'member-order': 'bysource'
}

# HTML

html_theme = 'sphinx_rtd_theme'
html_theme_options = {'logo_only': False}
templates_path = ['_templates']
html_static_path = ['_static', 'img']
html_logo = logo
html_short_title = '{} docs'.format(project)
html_copy_source = False
html_show_sphinx = False
html_use_index = False


# Setup


def setup(app):
    app.add_css_file('style.css')
