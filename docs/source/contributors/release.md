# Pull Requests & Releases

## Pull Requests

Use pull requests to add features, fix bugs, update docs, and merge work
between branches. These PRs should target a feature branch, not `main`, and
normally should not include a version bump.

Before merging any PR, verify:

1. API and docs are aligned:
   - docstrings and tutorials reflect current user-facing behavior,
   - top-level exports and user-facing interfaces match documentation,
   - official tutorial notebooks have been re-run and committed with
     up-to-date outputs when their behavior or output changed (see
     [documentation](documentation.md) for notebook policy).
2. Quality gates pass:
   - `make test` (default test suite),
   - `make test-perf` (performance degradation checks) for major changes, especially changes to the model or optimization code,
   - `make lint` (ruff checks),
   - `make docs-strict` (docs build without warnings).

## Release Process

Release PRs collect the changes intended for a release and target `main`. They
include the release metadata, including the version bump. Before merging a
release PR, verify the pull request checklist above and also check:

1. Version is updated consistently (`pyproject.toml`, docs metadata like `docs/source/conf.py`).
2. `CHANGELOG.md` is updated with user-facing changes (features, fixes, breaking changes, deprecations).
3. Deprecations are documented and tested where user-facing behavior changes.
4. Distribution artifacts are validated:
   - run `poetry build` and confirm sdists and wheels are created under `dist/`.

After the release PR is merged into `main`, tag the merge commit which will trigger the PyPI publication CI.
```bash
git switch main
git pull --ff-only
VERSION=$(poetry version -s)
TAG="v${VERSION}"
git tag "${TAG}"
git push origin "${TAG}"
```

Note: If the PR has been merged to the privat instance of the repo (for testing and prototyping) mirror it to the release remote (here called `public_ggml`):

```bash
git push public_ggml main
git push public_ggml "${TAG}"
```
