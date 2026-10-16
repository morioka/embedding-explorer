# Repository Guidelines

## Project Structure & Module Organization

- `app.py` is the Streamlit UI and local application entry point.
- `explorer/` contains reusable Python modules for vector normalization, PCA/UMAP projection, ingestion, SQLite/Vec1 persistence, caching, and optional EmbeddingGemma inference.
- `tests/` contains pytest unit and persistence tests.
- `scripts/` provides data workflows: `make_demo.py` creates synthetic data, `build_images.py` generates image embeddings, and `import_npz.py` imports or replaces an NPZ dataset.
- `data/` holds sample metadata, embeddings, images, SQLite state, and projection caches. Treat generated files as local data, not source code.

## Build, Test, and Development Commands

Create an environment and install runtime dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Generate the demo and run the UI:

```bash
PYTHONPATH=. python scripts/make_demo.py
streamlit run app.py
```

Run the test suite with `python -m pytest -q`. SQLite Vec1 tests run only when `VEC1_EXTENSION` points to a compiled official Vec1 library; do not substitute `sqlite-vec`.

## Coding Style & Naming Conventions

Use Python 3.11+ with four-space indentation, clear type/data validation, and small single-purpose functions. Follow PEP 8 naming: `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Keep imports standard-library, third-party, then local. No formatter or linter is configured, so preserve the surrounding style and run tests after edits.

## Testing Guidelines

Add focused tests under `tests/` using `test_*.py` files and `test_*` functions. Use `tmp_path` for filesystem/database fixtures and deterministic NumPy seeds for numerical tests. Cover both normal behavior and invalid dimensions/data where practical. UI rendering and actual model inference are not end-to-end tested.

## Commit & Pull Request Guidelines

Git history is not available in this checkout, so no repository-specific commit convention can be confirmed. Use concise imperative commit subjects (for example, `Add cached projection invalidation`). Pull requests should explain behavior changes, include test commands/results, identify data or schema migrations, and attach screenshots for Streamlit UI changes.

## Data and Security Notes

Do not mix synthetic demo vectors with model-generated vectors or datasets from different models. SQLite is authoritative after first import; use `scripts/import_npz.py --replace` for intentional replacement. Projection caches contain Python pickle files and must only be loaded from trusted local files. The app is intended for trusted local users, not an authenticated multi-tenant deployment.
