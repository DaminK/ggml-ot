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
poetry run pytest
```

Run with detailed output and warning handling:

```bash
poetry run python -W "ignore::DeprecationWarning:ot.*" -m pytest
```

Generate local HTML reports for test results and coverage:

```bash
poetry run pytest --cov=ggml_ot --cov-branch --cov-report=term-missing:skip-covered --cov-report=html --html=pytest-report.html --self-contained-html
```

This writes a self-contained pytest report to `pytest-report.html` and the
coverage site to `htmlcov/index.html`.

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
poetry run pytest -m "perf"
```

Update baselines:

```bash
poetry run pytest -m "perf" --update-baseline
```

Render the current performance overview as HTML:

```bash
poetry run python -m tests.utils.performance_snapshot --write-overview --write-overview-html
```

On GitHub Actions, the pytest workflow uploads these reports as job artifacts:

- self-contained pytest HTML
- coverage HTML for the mandatory tier
- performance overview HTML for the perf tier
- JUnit XML for machine-readable test results
