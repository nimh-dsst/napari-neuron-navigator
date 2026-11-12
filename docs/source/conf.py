"""Sphinx settings for the Neuron Navigator user documentation."""

from datetime import datetime

project = "napari-neuron-navigator"
author = "napari-neuron-navigator contributors"
copyright = f"{datetime.now().year}, {author}"  # noqa: A001

# Match Chronoxiv's user-documentation stack, with Markdown authoring added.
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.githubpages",
]
language = "en"
exclude_patterns: list[str] = []

html_theme = "pydata_sphinx_theme"
html_title = "Neuron Navigator"
html_static_path = ["_static"]
html_css_files = ["brand.css"]
# docs-build generates the PDF before Sphinx copies it into the published site.
html_extra_path = ["../_build/pdf"]
html_baseurl = "https://nimh-dsst.github.io/napari-neuron-navigator/"

html_theme_options = {
    "logo": {"text": "Neuron Navigator"},
    "github_url": "https://github.com/nimh-dsst/napari-neuron-navigator",
    "icon_links": [
        {
            "name": "Issue tracker",
            "url": "https://github.com/nimh-dsst/napari-neuron-navigator/issues",
            "icon": "fa-solid fa-bug",
        },
    ],
    "header_links_before_dropdown": 7,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
