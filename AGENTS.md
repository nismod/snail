# Repository Guidelines

## Project Structure & Module Organization
The core Python package lives in `src/snail/`, with the command-line interface exposed in `src/snail/cli.py` and subpackages for intersection, routing, and damages logic. Unit and integration tests sit in `tests/`, following `test_<feature>.py` files that mirror module names. The C++ raster intersection engine is under `extension/src/`; its CMake build outputs into `build/`. Sphinx docs and tutorial notebooks are in `docs/` (notably `docs/source/tutorials`), with sample datasets under `data/` and CLI workflows in `scripts/`.

## Build, Test, and Development Commands
Set up a clean environment (conda or venv) and run `pip install -e .[dev]` to install editable sources plus tooling. Execute `pytest` for the Python suite; append `--cov=snail --cov-report=term-missing` before opening a PR. For the C++ layer run `cmake -Bbuild ./extension && cmake --build build` and validate with `./build/run_tests`. Regenerate docs locally with `sphinx-build docs/source docs/build/html` if you touch tutorials.

## Coding Style & Naming Conventions
Python code is formatted with `black` (line length 79). Use snake_case for functions and variables, PascalCase for classes, and keep module names lowercase. Prefer typing hints for new public APIs and raise `ValueError` or custom errors for invalid inputs. When touching the C++ library, run `clang-format -i extension/src/*.{cpp,hpp}` and keep header guards consistent with existing files.

## Testing Guidelines
Add targeted pytest cases beside the feature under test, using fixtures for geospatial datasets (see `tests/core`). Name tests `test_<behavior>` and keep assertions explicit. Maintain or improve coverage, especially around grid slicing and routing paths. C++ changes require extending the Catch2 suite before calling `./build/run_tests`.

## Commit & Pull Request Guidelines
We follow concise, imperative commit subjects (`Fix pypi link`, `Merge artifacts dirs`). Use body text for context, referencing issue numbers with `Fixes #123` where applicable. Pull requests should summarize the change, highlight data or CLI impacts, link related issues, and include screenshots or sample outputs when UI-facing tutorials change. Mention required follow-up tasks and note any skipped tests.
