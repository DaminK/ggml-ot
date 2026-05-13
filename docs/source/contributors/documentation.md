# Documentation

A good documentation is essential. You can contribute by improving existing docs or adding new material.

## 1. Docstrings

Every public function and class should have a docstring. We use [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings parsed by [sphinx.ext.napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html). The API reference is built automatically with [Sphinx](https://www.sphinx-doc.org/) and hosted on [Read the Docs](https://ggml-ot.readthedocs.io/en/latest/).

You can create cross-references between docstrings using Sphinx roles. For example:

```python
def add(a, b):
    """Add two numbers.

    See :func:`sum` for a more general function.
    See :class:`Addition` for more information.

    Parameters
    ----------
    a : float
        First number.
    b : float
        Second number.

    Returns
    -------
    float
        The sum of *a* and *b*.
    """
    return a + b
```

## 2. Sphinx documentation

We use [MyST Parser](https://myst-parser.readthedocs.io/) for Markdown (`.md`) and reStructuredText (`.rst`) support. With [nbsphinx](https://nbsphinx.readthedocs.io/), we also include Jupyter notebooks (`.ipynb`) directly in the docs.

The documentation lives in `docs/`. Tutorial notebooks are in `docs/source/tutorials/`.

When contributing tutorials:

- Provide clear explanations and code examples.
- Run the notebook locally before committing and commit it with up-to-date outputs.
- Do not clear outputs before committing if they are needed for the reader to follow the tutorial.
- Notebooks are not re-executed automatically in CI. The committed output is what appears in the published docs.

### Integration tutorials (external packages)

We explicitly welcome tutorials that show how to use `ggml-ot` together with other Python packages, especially from the scverse ecosystem (for example Scanpy/scvi-tools-style workflows).

For integration tutorials, include:

- The target package(s) and intended user workflow.
- The handoff points between tools (inputs/outputs, data structures, key fields).
- A minimal reproducible example that users can run end-to-end.
- Notes on limitations, version assumptions, or optional dependencies.

To ensure docs build correctly, install docs dependencies and run the strict docs build:

```bash
poetry install --with docs
make docs-strict
```

`make docs-strict` runs `sphinx-build -W` which treats Sphinx warnings as errors, while `make docs` ignores warnings. To re-execute all tutorial notebooks during the build, you can use `make docs-with-nbsphinx`  (very slow, only for major changes).

For a quick local preview while editing, serve the built docs:

```bash
python -m http.server 8000 -d docs/build/html
```

In VS Code Remote-SSH, this usually triggers a forwarded-port popup so you can open the docs in your browser directly.
