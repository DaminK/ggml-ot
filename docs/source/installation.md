# {octicon}`plug;1em;` Installation

There are different ways to install ggml-ot, depending on your needs. It is recommended to install ggml-ot inside a virtual environment to avoid dependency conflicts. You can create one using [venv](https://docs.python.org/3/library/venv.html), [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) or [poetry](https://python-poetry.org/docs/managing-environments/)

## Pip/PyPI

The easiest way to get started is to install the latest release from PyPI:
```bash
   pip install ggml-ot
```

## Development Version

If you want to use the latest version directly from GitHub, you can clone the repository and install with Poetry:
```bash
   git clone https://github.com/DaminK/ggml-ot
   cd ggml-ot
   pip install poetry
   poetry lock && poetry install
```
This will install the package in a regular environment without developer tools.

For contributing to ggml-ot, you can install it in different modes, see [the contributor guide](contributors_guide.md#installation-of-development-version) for more details.
