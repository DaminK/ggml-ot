# Testing

We use [pytest](https://docs.pytest.org/en/stable/) for automated tests.

## Writing tests

- Place tests in `tests/`.
- Name files `test_*.py`.
- Name test functions `test_*`.
- For each new public feature, add tests that verify expected user-facing behavior.

## Test design principles

- Test documented **public behavior** and user-facing contracts.
- Prefer behavior assertions over implementation details (internal helpers, call chains, exact internal wiring).
- Avoid "removed API stays removed forever" checks unless there is an explicit long-term compatibility policy.
- Before `1.0`, avoid accumulating alias/legacy regression tests for short-lived API transitions.
- Keep warning/error tests close to their module tests and assert stable message fragments rather than brittle full strings.

## Running tests

Run the full suite locally:

```bash
make test
```

This runs `poetry run pytest`.

For small changes, `make test` is usually sufficient. For major changes,
especially changes to the model or optimization code, also run `make test-perf`
to check for performance degradation across the benchmark setups.

Run with detailed output and warning handling:

```bash
poetry run python -W "ignore::DeprecationWarning:ot.*" -m pytest
```

Generate a local coverage report:

```bash
make coverage
```

This runs `poetry run pytest --cov=ggml_ot --cov-report=term-missing
--cov-report=html` and writes the coverage site to `htmlcov/index.html`. Use a
direct `poetry run pytest ...` command if you need extra report formats.

## Network vs synthetic data tests

Some tests can optionally download real-world datasets.

Enable network tests:

```bash
poetry run pytest -m "network"
```

Force synthetic-only tests:

```bash
poetry run pytest -m "not network"
```

## Performance tests

Performance benchmarks are marked `perf` and excluded by default.

Run benchmarks:

```bash
make test-perf
```

This runs `poetry run pytest -m perf --data-source "all" --update-baseline`.

Render the current performance overview as HTML:

```bash
poetry run python -m tests.utils.performance_snapshot --write-overview --write-overview-html
```

On GitHub Actions, the pytest workflow uploads these reports as job artifacts:

- self-contained pytest HTML
- coverage HTML for the mandatory tier
- performance overview HTML for the perf tier
- JUnit XML for machine-readable test results
