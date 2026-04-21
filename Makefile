.PHONY: help test perf coverage lint docs docs-fast clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:  ## Run default test suite (excludes perf and dev)
	poetry run pytest

perf:  ## Run performance benchmarks
	poetry run pytest -m perf --data-source "all" --update-baseline

test-all:  ## Run all tests including perf and dev
	poetry run pytest -m "" --override-ini="addopts="

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

coverage:  ## Run tests with coverage report (terminal + HTML in htmlcov/)
	poetry run pytest --cov=ggml_ot --cov-report=term-missing --cov-report=html

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:  ## Run ruff linter
	poetry run ruff check --config pyproject.toml

format:  ## Auto-format with ruff
	poetry run ruff format --config pyproject.toml

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

docs:  ## Build Sphinx docs (HTML, skips notebook execution)
	MPLCONFIGDIR=/tmp/mplconfig poetry run sphinx-build -b html -D nbsphinx_execute=never docs/source docs/build/html

docs-full:  ## Build Sphinx docs with notebook execution
	MPLCONFIGDIR=/tmp/mplconfig poetry run sphinx-build -b html docs/source docs/build/html

docs-strict:  ## Build Sphinx docs with -W (warnings as errors)
	MPLCONFIGDIR=/tmp/mplconfig poetry run sphinx-build -b html -W -D nbsphinx_execute=never docs/source docs/build/html

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove build artifacts
	rm -rf docs/build htmlcov reports .coverage .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
