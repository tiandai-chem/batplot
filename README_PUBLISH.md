Publishing to PyPI (macOS, zsh)

This guide shows how to publish a new version of `batplot` to TestPyPI (recommended) and PyPI. Commands assume you’re in the workspace root and using a virtual environment.

## 0) Activate venv and tools
```bash
cd "/Users/tiandai/Library/CloudStorage/OneDrive-UniversitetetiOslo/My files/batplot_script"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip build twine
```

## 1) Bump the version
- Edit `pyproject.toml` → `[project].version` (must be new on PyPI).
- Keep the fallback in `batplot/__init__.py` (`__version__`) in sync.

## 2) Quick sanity check
```bash
python -m batplot.cli --help
```

## 3) Build artifacts (sdist + wheel)
```bash
python -m build
```
Artifacts will appear in `dist/`.

## 4) Validate artifacts
```bash
python -m twine check dist/*
```

## 5) Upload to TestPyPI (dry run in the wild)
- Create a token at: https://test.pypi.org/manage/account/#api-tokens
- Export credentials (username is literal `__token__`):
```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-AgENdGVzdC5weXBpLm9yZy4uLi"   # replace with your token
```
- Upload:
```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
- Verify install from TestPyPI in a clean venv:
```bash
python -m venv /tmp/bp-test
source /tmp/bp-test/bin/activate
python -m pip install -U pip
python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple batplot==<new-version>
python -m batplot.cli --help
deactivate
```

## 6) Upload to PyPI
- Create a token at: https://pypi.org/manage/account/#api-tokens
- Export credentials:
```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-AgENdHlwaS5vcmcuLi4"   # replace with your token
```
- Upload:
```bash
python -m twine upload dist/*
```

## Troubleshooting
- HTTP 400 "File already exists": bump the version in `pyproject.toml` and rebuild.
- "No module named build": `python -m pip install -U build` inside your venv, then re-run build.
- Paths with spaces (OneDrive): quote the interpreter in tasks, e.g. `"./.venv/bin/python" -m build`.
- CLI script glitches in some venvs: invoke via module `python -m batplot.cli`.
- Missing long description on PyPI: ensure `README.md` is present and referenced by `readme = "README.md"` in `pyproject.toml`.

## Project metadata
- Backend: setuptools (PEP 517 / PEP 621 in `pyproject.toml`).
- Console script: `batplot` → `batplot.cli:main`.
- URLs: update `[project.urls]` when repository/homepage moves.
