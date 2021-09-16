import os
import sys

sys.path.extend([
    os.path.abspath('..'),
    os.path.abspath('../lib/pyutils')
])

from evowluator.data.syntax import Syntax

# Project metadata

project = 'evOWLuator'
copyright = '2019-2020, SisInf Lab'
author = 'SisInf Lab'
version = '0.1.2'
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

syntaxes = ', '.join(f':code:`{s}`' for s in Syntax.all())
rst_prolog = f':github_url: {git_url}'
rst_epilog = (
    f'.. |syntaxes| replace:: {syntaxes}\n'
    f'.. _git_url: {git_url}\n'
)

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
html_short_title = f'{project} docs'
html_copy_source = False
html_show_sphinx = False
html_use_index = False


# Setup


def setup(app):
    app.add_css_file('style.css')
