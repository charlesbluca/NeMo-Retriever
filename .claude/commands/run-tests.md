Reproduce a CI check locally. Maps directly to the checks in `.github/workflows/`.

TRIGGER when: user asks to run tests, run pytest, check CI, run pre-commit, or verify a change before committing or opening a PR.
SKIP: harness benchmarks (use harness-run skill instead).

## Step 1 — identify context

Check whether the user is on a PR branch:
```bash
git branch --show-current
gh pr status 2>/dev/null || true
```

If on a PR branch (or the user mentions PR prep), note that the full PR CI (`ci-pull-request.yml`) requires **both** Pre-commit Checks and Legacy Docker Build & Test to pass, and offer to run them in sequence.

## Step 2 — pick a category

Present the three CI categories and ask which one to reproduce:

**A. Retriever Unit Tests** (`retriever-unit-tests.yml`)
- Runs on every branch push, independently of PR/main CI
- Fast Python-only; no Docker needed
- Local equivalent (must run from `nemo_retriever/` subdir using `uv`):
  ```bash
  cd nemo_retriever && PYTHONPATH=src uv run python -m pytest tests -q
  ```
- **Test layout**: harness tests are flat files directly in `nemo_retriever/tests/` (e.g. `test_harness_config.py`, `test_harness_run.py`), not in a `tests/harness/` subdirectory. When a user refers to "harness tests" or passes `nemo_retriever/tests/harness/`, translate to the correct pattern:
  ```bash
  cd nemo_retriever && PYTHONPATH=src uv run python -m pytest tests/test_harness_*.py -q
  ```
- For any other path like `nemo_retriever/tests/foo/bar.py`, strip the `nemo_retriever/` prefix and run from the subdir: `cd nemo_retriever && PYTHONPATH=src uv run python -m pytest tests/foo/bar.py -q`

**B. Pre-commit Checks** (part of `ci-pull-request.yml` and `ci-main.yml`)
- Runs: black, flake8, trailing-whitespace, end-of-file-fixer, check-ast, debug-statements, validate-deployment-configs
- Local equivalent:
  ```bash
  pre-commit run --all-files
  ```
- To re-run only against staged/changed files: `pre-commit run`

**C. Legacy Docker Build & Test** (part of `ci-pull-request.yml` and `ci-main.yml`)
- Builds the image with `--target test`, then runs the legacy service/client/api test suites inside the container
- Test paths inside container: `tests/service_tests client/client_tests api/api_tests`
- Marker filter: `not integration`
- Local build + test (mirrors `reusable-docker-build-and-test.yml`):
  ```bash
  # Build
  docker buildx build --load --target test -t nv-ingest:local .

  # Run tests
  docker run --rm nv-ingest:local bash -lc \
    "python -m pip install --disable-pip-version-check 'pymilvus[milvus_lite]' && \
     python -m pytest -rs -m 'not integration' tests/service_tests client/client_tests api/api_tests"
  ```
- If an image is already built locally, skip the build step and run tests directly against it.
- Note: this exercises the **deprecated** `src/`, `api/`, `client/` packages, not `nemo_retriever/`.

## PR prep shortcut

If the user is prepping a PR and wants to mirror exactly what CI will gate on, run B then C in sequence (Retriever Unit Tests run automatically on push and don't gate the PR merge, but are still worth checking).
