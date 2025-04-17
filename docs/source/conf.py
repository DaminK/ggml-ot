import os

project = "ggml-ot"
author = "Damin Kuehn <damin.kuehn@rwth-aachen.de>"
release = "0.0.1"

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
html_theme = "sphinx_rtd_theme"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx_autorun",
]
