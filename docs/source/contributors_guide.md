# {octicon}`code-of-conduct;1em;` Contributor’s Guide

Thank you for considering contributing to **ggml-ot**.
There are different ways to contribute — from fixing typos to proposing and adding new features. This document describes how you can get involved.

---

## What to Contribute

We welcome contributions in many forms, including:

- **Code**
   - Proposing and implementing **new features** and existing suggestions in the **issues**.
   - Finding, reporting and fixing **bugs**.
   - Improving existing code.
- **Documentation**
   - Adding and improving **documentation** to (new) features.
   - Adding and updating **tutorials** and **examples**.
- **Tests**
   - Implementing **tests** for existing features.
- **Ideas**
   - Suggesting **new functionalities** and **improvements**.
   - Sharing **feedback** and proposing **design changes**.

---

## Installation of Development Version

If you want to use the latest version directly from GitHub, you can clone the repository and install with Poetry:
```bash
   git clone https://github.com/DaminK/ggml-ot
   cd ggml-ot
   pip install poetry
   poetry lock && poetry install
```
This will install the package in a regular environment without developer tools.

For contributing to ggml-ot, you can install it in different modes.
For development contribution, use `--with dev` with `poetry install`, for documentation contribution, use `--with docs`, and for testing, use `--with test`.
To set up the pre-commit hooks, run `pre-commit install`.

---

## Code-Style and Organisation

When writing code, please follow general principles for clean and maintainable code:
- Write small and **modular** functions focusing on a single task.
- Use **meaningful names** for variables, functions, and classes.
- For every public function and class, write a clear **docstring** following the [Sphinx format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html). Keep the docstrings **precise** and focused on what the code does, its arguments, and return values.
- Add **short inline** comments for complex logic and **Python type hints**.
- Avoid **duplicate code** by refactoring it into **helper functions**.
- Do not forget to add new (public) functions to the corresponding `_init_.py` file.

When implementing new functionalities, please make sure to add new dependencies to the `pyproject.toml` file. You can do this by running the following command:
```bash
   poetry add <package-name>
```
If you want to add a dependency only to a specific group, e.g. the documentation group, use `--group docs` after the package name.
You can remove dependencies by running `poetry remove <package-name>`.

We use [pre-commit](https://pre-commit.com/) hooks to enforce a consistent style across the project. This ensures that code is well-formatted, linted, and free from common issues before it is committed. If you followed the development installation instructions, hooks run automatically on every commit. If a hook fails, fix the issues and re-stage your files.

You can also run all hooks manually:
```bash
   pre-commit run --all-files
```

Keep the folder and file structure in mind when implementing new features. Write contributions of new functionalities in the folder `ggml_ot/`. Place functions regarding the training of the data in `ggml.py`, functions regarding the evaluation of trained models in the `benchmark/` folder, functionalities that include processing data and generating datasets into `data/`, new functions concerned with computing distances into `dists/`, and functions regarding visualization into the `plot/` folder. <br>
In the folder `examples/`, we collect Jupyter notebooks that show how to use the implemented functions. <br>
Everything concerning the documentation of the package is located in the folder `docs/`. <br>
Lastly, we collect the pytests in the folder `tests/`.

---

## Writing Documentation

A good documentation is very important. You can contribute by writing or improving documentation to existing code or adding new documentation.
There are two main ways to contribute:

### 1. Docstrings

Every public function and class should have a docstring. We use the [Sphinx format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) for the docstrings which are automatically included in the API reference using [Sphinx](https://www.sphinx-doc.org/). Our documentation is hosted on [Read the Docs](https://ggml-ot.readthedocs.io/en/latest/).
You can create links between docstrings to refer from one function or class to another. For example, to refer from `add` to the function `sum` or to the class `Addition`, you can use the following syntax:

```python
def add(a, b):
    """
    Add two numbers.
    See :func:`sum` for a more general function.
    See :class:`Addition` for more information.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: The sum of a and b
    :rtype: float
    """
    return a + b
```

### 2. Sphinx Documentation

We use [MyST Parser](https://myst-parser.readthedocs.io/) to allow both Markdown (`.md`) and reStructuredText (`.rst`) files. Using [nbsphinx](https://nbsphinx.readthedocs.io/), we also include Jupyter Notebooks (`.ipynb`) directly in the documentation. The documentation lives in the `docs/` folder. Tutorial notebooks are collected in `docs/source/tutorials/` and correspond to the examples in the `examples/` folder.
They can be slightly adapted by for example adding links to external websites or other Jupyter Notebooks:

```bash
   ### Add to markdown cell or file
   [For more details, checkout the Tutorial on synthetic data](ggml_synth_tutorial.html)
```

When contributing tutorials, please:
- Provide simple and clear **explanations** and **code examples**.
- Do not clear the **notebook output** before committing and keep it minimal - only what is necessary to understand the example.
- Run notebooks locally before committing them to ensure they execute without errors.

To make sure the documentation builds correctly, you can build the docs locally using Sphinx directly. Before doing so, make sure to install the needed dependencies with `poetry install --with docs`:
```bash
  python -m sphinx -T -W --keep-going -b html -d docs/build/doctrees docs/source docs/build
```
The flag `-T` prints the full Python traceback on errors and `-W --keep-going` treats warnings as errors but keeps processing other documents even if some fail, so a complete error list is printed instead of stopping at the first failure. With `-b html` the output type is specified as HTML, `-d docs/build/doctrees` sets the doctree directory and `docs/source docs/build` specifies where to read from and write to.

---

## Pytests

We use [pytest](https://docs.pytest.org/en/stable/) to run all automated tests in this project. Running tests ensures that changes do not break existing functionality.

### Writing Tests

The tests are located in the `tests/` folder. Test files should be named `test_*.py` and test functions inside those files should be named `test_*()`. For every new functionality, please write a test to ensure that it works as expected. Make sure all tests pass locally before submitting the changes.

### Running Tests

On every push and pull request, GitHub Actions lints the code with [ruff](https://docs.astral.sh/ruff/), runs the tests on supported Python versions, and saves the results for review.
This ensures that contributions are checked for compatibility and correctness before being merged.
If you want to run the tests locally, you can use the following command:
```bash
   poetry run pytest
```

This runs pytest inside the Poetry-managed environment. If you want to see detailed output and warnings, you can run:
```bash
   poetry run python -W "ignore::DeprecationWarning:ot.*" -m pytest
```

---

## Issues

We manage open tasks, planned features, and bugs in the [issues](https://github.com/DaminK/ggml-ot/issues) section of the GitHub repository.
You can work with the issues in different ways:

### Writing an Issue

If you encounter a bug in the code, a mistake in the documentation or want to propose a new feature, please open an issue on GitHub. Before doing so, please check if there is already an issue for your problem and if so, feel free to add your comments to the existing issue instead of creating a new one.
Make sure to precisely formulate your request. You can add labels to the issue to categorize it and add milestones to mark its importance and priority.

### Working on an Issue

If you want to contribute but do not know where to start, check the [existing issues on GitHub](https://github.com/DaminK/ggml-ot/issues).
Look at the issue labels to get an idea of what is needed.

If you would like to work on an issue, please comment on it and assign yourself first. This lets others know that someone is already working on it and helps avoid duplicate work. In this way, we keep contributions coordinated and communication clear.

---

## Contribution Process

If you want to contribute in any way to the code or documentation of the project, please follow the steps below.

1. **Fork** the repository to your own GitHub account.
2. **Clone** your fork locally.
3. Create a **new branch** for your contribution. Make sure to use a descriptive name for your branch by e.g. referencing the issue.
4. Make your changes and respect the guidelines described in this document. Please test your changes thoroughly before committing them.
5. After pushing your changes to your fork, open a **pull request** to the main branch of the original repository. In your pull request, please provide a clear description of your changes and the reasoning behind them.

Once your pull request is finished, the maintainers will review it and merge it if it is a reasonable improvement to the codebase.

---

Thank you for your contributions to **ggml-ot**!

---

## Contact

If you have any questions or suggestions, feel free to contact us
<a href="mailto:kuehn@cs.rwth-aachen@@de?subject=ggml_ot:" onclick="this.href=this.href.replace('@@','.')"> via mail</a>.
