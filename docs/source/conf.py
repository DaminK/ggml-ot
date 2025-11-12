import os

project = "ggml-ot"
author = "Damin Kuehn <damin.kuehn@rwth-aachen.de>"
repository_url = "https://github.com/DaminK/ggml-ot"
release = "0.9.9"

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
html_theme = "scanpydoc"
html_logo = "images/icon_ggml_ot.png"
html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": "main",
    "use_repository_button": True,
    "navigation_depth": 4,
}

master_doc = "index"
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx_autorun",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "scanpydoc",
    "sphinx_design",
    "sphinx.ext.linkcode",
]

suppress_warnings = ["nbsphinx.localfile"]

autosummary_generate = True

myst_heading_anchors = 3
myst_enable_extensions = ["html_image"]

nbsphinx_thumbnails = {
    "tutorials/ggml_cross_validation": "_images/cross_validation.png",
}
