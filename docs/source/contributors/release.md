# Release Process

## Release checklist

Before creating a new release, verify:

1. Version is updated consistently (`pyproject.toml`, docs metadata like `docs/source/conf.py`).
2. `CHANGELOG.md` is updated with user-facing changes (features, fixes, breaking changes, deprecations).
3. Deprecations are handled explicitly:
   - user-visible deprecation warnings are added where needed,
   - removal timelines are documented,
   - corresponding tests are added or updated.
4. Public API and docs are aligned:
   - docstrings and tutorials reflect current public behavior,
   - top-level exports and user-facing interfaces match documentation,
   - official tutorial notebooks have been re-run and committed with up-to-date outputs (see [documentation](documentation.md) for notebook policy).
5. Quality gates pass:
   - `poetry run pytest` (full test suite),
   - `poetry run ruff check` (lint and format),
   - `poetry run sphinx-build -b html -W --keep-going docs/source docs/build` (docs build, zero warnings).
6. Distribution artifacts are validated:
   - run `poetry build` and confirm sdists and wheels are created under `dist/`.

## CI publishing model

The publish workflow follows this pattern:

- On normal branch pushes and pull requests: build-only check (`poetry build`).
- On version-tag pushes: build and publish the produced `dist/*` artifacts to PyPI.

This keeps release publishing strict while keeping regular CI lightweight.
