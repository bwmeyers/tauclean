# Building the Documentation

This directory contains the Sphinx documentation for tauclean.

## Prerequisites

Install the documentation dependencies:

```bash
uv sync --group docs
```

Or with pip:

```bash
python -m pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

## Building

To build the HTML documentation:

```bash
uv run sphinx-build -b html . _build/html
```

The built documentation will be in `_build/html/index.html`.

### Other build targets

- `make clean` - Remove all build artifacts
- `make latex` - Build LaTeX files
- `make latexpdf` - Build PDF documentation (requires LaTeX)
- `make epub` - Build EPUB format
- `make linkcheck` - Check all external links

## Auto-rebuilding (development)

To automatically rebuild the documentation when source files change, you can use `sphinx-autobuild`:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

This will start a local server at http://127.0.0.1:8000.

## Documentation Structure

- `index.rst` - Main documentation page
- `getting_started.rst` - Getting started guide
- `api/` - API reference documentation
- `references.rst` - Bibliography and references
- `conf.py` - Sphinx configuration
