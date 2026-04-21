# Development Setup

## Installation for Contributors

Clone the repository and install with Poetry:

```bash
git clone https://github.com/DaminK/ggml-ot
cd ggml-ot
pip install poetry
poetry lock && poetry install
```

This installs the package without extra contributor tooling.

### Recommended installs (by task)

- **Development (code + tests + docs)**

  ```bash
  poetry install --with dev,test,docs
  ```

- **Development only** (code + tooling)

  ```bash
  poetry install --with dev
  ```

- **Testing only**

  ```bash
  poetry install --with test
  ```

- **Docs only** (Sphinx + notebooks)

  ```bash
  poetry install --with docs
  ```

We keep `dev`, `test`, and `docs` separated so users can install only what they need.
For contributors, the combined install is usually the most convenient.

To set up pre-commit hooks, first ensure the `dev` group is installed, then run:

```bash
pre-commit install
```

### Quickstart for contributors

```bash
poetry install --with dev,test,docs
pre-commit install
make test
make docs
```

## Makefile

Common tasks are available as `make` targets. Run `make help` to list them:

| Target         | Description                                       |
|----------------|---------------------------------------------------|
| `make test`    | Run default test suite (excludes perf and dev)    |
| `make perf`    | Run performance benchmarks                        |
| `make test-all`| Run all tests including perf and dev markers      |
| `make coverage`| Run tests with coverage report (terminal + HTML)  |
| `make lint`    | Run ruff linter                                   |
| `make format`  | Auto-format with ruff                             |
| `make docs`    | Build Sphinx docs (skips notebook execution)      |
| `make docs-full`| Build Sphinx docs with notebook execution        |
| `make docs-strict`| Build Sphinx docs with warnings as errors      |
| `make clean`   | Remove build artifacts and caches                 |

### Environment notes

- Supported Python versions: see `pyproject.toml` (`requires-python`).
- GPU/CUDA is optional; most tests run on CPU.
- If you use a non-default Python, set it via `poetry env use /path/to/python`.

## Code-Style and Organisation

When writing code, please follow these principles:

- Write small and **modular** functions focusing on a single task.
- Use **meaningful names** for variables, functions, and classes.
- For every public function and class, write a clear **docstring** following the [NumPy-style format](https://numpydoc.readthedocs.io/en/latest/format.html) (parsed by `sphinx.ext.napoleon`).
- Add short inline comments for complex logic and use Python type hints.
- Avoid duplicate code by refactoring into helper functions.
- Add new public functions to the corresponding `__init__.py` file.

When implementing new functionality, add dependencies to `pyproject.toml`:

```bash
poetry add <package-name>
```

To add a dependency only to a specific group (for example docs):

```bash
poetry add --group docs <package-name>
```

To remove dependencies:

```bash
poetry remove <package-name>
```

We use [pre-commit](https://pre-commit.com/) hooks to enforce a consistent style across the project.
You can run all hooks manually:

```bash
pre-commit run --all-files
```

Keep the folder and file structure in mind when implementing new features:

- `ggml_ot/`: package implementation
- `ggml_ot/benchmark/`: evaluation and cross-validation
- `ggml_ot/data/`: data processing and dataset generation
- `ggml_ot/distances/`: distance and OT computation
- `ggml_ot/gene/`: gene ranking and enrichment analysis
- `ggml_ot/gmm/`: Gaussian mixture model fitting
- `ggml_ot/optimization/`: training loop and loss functions
- `ggml_ot/plot/`: visualization code
- `docs/`: documentation sources and tutorial notebooks
- `tests/`: pytest suite
