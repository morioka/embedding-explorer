# Embedding Explorer

Streamlit + Plotly + PCA/UMAP + NumPy/SQLite Vec1. Fully local UI. Image model is optional and lazy-loaded.

## 1. Install (Ubuntu / WSL2, Python 3.11 or 3.12 recommended)

```bash
cd embedding-explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The verified local setup in this checkout uses pyenv's `work` environment instead:

```bash
pyenv activate work
python -m pip install -r requirements.txt
```

Alternative with uv: `uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`.

## 2. Demo without any model download

```bash
PYTHONPATH=. python scripts/make_demo.py
streamlit run app.py
```

The demo contains 72 generated graphic thumbnails and **synthetic 48-dimensional clustered vectors**. They are not image embeddings. Choose **Existing sample** as the query source, then switch PCA/UMAP, k, thumbnails, and NumPy/SQLite Vec1 to verify the pipeline.

## 3. Use existing embeddings

Prepare `data/embeddings.npz` with:

- `vectors`: float32 (N,D), finite, nonzero, all same dimension.
- `labels`: N strings.
- `paths`: N strings, paths of image files relative to data directory (e.g. `images/cat.png`).

Example:

```python
import numpy as np
np.savez_compressed('data/embeddings.npz', vectors=vectors.astype('float32'), labels=np.array(labels), paths=np.array(paths))
```

Optional `data/manifest.json`: `{"model_id":"some-model-revision", "dimension":768}`.

**Query upload**: `.npy` containing one D-vector (or `.npz` with `vector` key). Use the same model, preprocessing and dimensionality as the sample vectors. Dataset is L2 normalized internally.

## 4. Actual images with EmbeddingGemma 2

The model used by this project is [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2).
It produces 768-dimensional multimodal embeddings and supports text and images in the same
embedding space. The project does not automatically fetch a model from the Internet: download
the complete model directory first, including tokenizer, image processor, pooling, configs and
weights.

Install the model dependencies in `work`:

```bash
pyenv activate work
pip install -r requirements-model.txt
```

Download the model from Hugging Face:

```bash
pip install -U huggingface_hub
mkdir -p models
hf download google/embeddinggemma-2 \
  --local-dir "$PWD/models/embeddinggemma-2"
```

Verify that the directory is complete before using it:

```bash
test -f models/embeddinggemma-2/config.json
test -f models/embeddinggemma-2/tokenizer.json
find models/embeddinggemma-2 -maxdepth 1 -type f | sort
```

Alternatively, download it with Python:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/embeddinggemma-2",
    local_dir="models/embeddinggemma-2",
)
PY
```

Use the downloaded directory for image indexing and the Streamlit sidebar:

```bash
# put at least 3 .png/.jpg/.webp files in data/gemma2/images/
PYTHONPATH=. python scripts/build_images.py \
  --model "$PWD/models/embeddinggemma-2" --data data/gemma2
streamlit run app.py
```

Set the Streamlit **Data directory** to `data/gemma2` and **Local model directory** to the same
absolute model path to enable query image upload and cross-modal text search.
`SentenceTransformer(..., local_files_only=True)` prevents implicit model fetches. A working
PyTorch installation appropriate to the machine is required; CPU inference is possible but slow.
The bundled `data/` directory is the 48-dimensional synthetic demo and must not be reused for
the 768-dimensional Gemma2 dataset.

## 5. Design details

- The projector fits to **corpus only**. Query uses `transform()`; it does **not refit** the embedding map.
- UMAP with cosine metric and PCA both operate on unit-normalized vectors.
- k-NN searches **original vectors**. Similarity uses `1 - cosine_distance`. SQLite Vec1 is configured as `flat` with cosine distance.
- SQLite is authoritative; NPZ is imported only if the DB is empty, or explicitly replaced. Batch uploads append transactionally and rebuild the Vec1 flat index from all SQLite embeddings after each batch.
- Sample points use Scattergl; thumbnails use Plotly layout images and are capped. Overlay images may intercept point clicks; turn thumbnails off to select points easily.
- Distinct model spaces, including synthetic vectors, must never be mixed in one search index. Data provenance should be verified externally.
- Projection geometry is an approximation, especially for a newly transformed query in UMAP; 2D distances and nearest-neighbor ranking need not agree.
- The `data_dir` sidebar input is for locally trusted users. This is not an authenticated multi-tenant service.

## 6. Checks

```bash
pip install pytest
python -m pytest -q
```

Core/projection/sqlite unit tests are provided. Streamlit UI and actual EmbeddingGemma 2 inference are not end-to-end tested here.

## References

- https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
- https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers
- https://huggingface.co/google/embeddinggemma-2
- https://sqlite.org/vec1/doc/trunk/doc/vec1.md
- https://umap-learn.readthedocs.io/en/latest/transform.html

## SQLite-first persistence (updated design)

`data/vectors.db` is the authoritative source for sample labels, image paths and float32
vectors. `sample_vectors` is the official SQLite Vec1 `vec1(vector)` index configured for flat cosine NN. On first
run, an existing `embeddings.npz` is imported **only if the database is empty**.
Subsequent launches load sample data directly from SQLite, ignoring NPZ changes.
To intentionally replace samples after generating a new NPZ:

```bash
PYTHONPATH=. python scripts/import_npz.py --data data --replace
```

PCA/UMAP training coordinates are persisted in the SQLite `projections` table,
keyed by dataset revision and projection settings. The trained transformer is
saved to `data/projection_models/<key>.pkl`; both must exist for reuse. A missing
or incompatible cache should be deleted to retrain. The first run with each
method/parameter combination trains and persists both. New query vectors always
use the fitted model's `transform()`, never refit the sample projection.

**Security:** projection models use Python pickle, which must never be loaded
from untrusted sources. Only use local projection cache files created by this
application. SQLite stores vector arrays and projection coordinates, so no
pickle deserialization is needed for those.

Changing samples via `import_npz.py --replace` increments the dataset revision
and invalidates old projection keys. Deleting the cache directory or the rows
in `projections` for an active key forces retraining.

## Batch image upload in Streamlit

1. Start with an **empty data directory** for the selected model, such as `data/my_photos`; choose it in the sidebar. The bundled `data/` directory contains synthetic demo vectors and **cannot** be mixed with genuine image vectors.
2. Set **Local model directory** to the complete, locally available EmbeddingGemma 2 model directory; install `requirements-model.txt` first.
3. Drop multiple PNG/JPEG/WebP images into **Add sample images (batch)**, then click **Embed, store and reproject**.
4. The app decodes images, appends embeddings to SQLite `samples`, rebuilds the Vec1 index from all SQLite embeddings, increments the dataset revision, refits the selected PCA/UMAP projector, persists the training coordinates in SQLite and the fitted transform in `projection_models/`, and redraws. Other projection methods are lazily re-fitted on next selection.
5. Duplicate uploads are ignored by SHA-256 content hash (same bytes). Files are copied into `images/uploaded/`. A model provenance check prevents appending to data created by a different model path. Existing vectors with no provenance cannot be mixed safely; use another data directory.

Embedding inference runs synchronously in the Streamlit process. This is intentionally a simple local workflow, not a background job queue; for very large uploads, progress reporting and job processing should be added. Closing/restarting the process during a batch may leave unused image files, but will not add partially embedded entries to SQLite. The UI does not automatically execute inference just because files were dropped: the explicit button prevents repeated processing when Streamlit reruns.

## Official SQLite Vec1 installation (required; not sqlite-vec)

Vec1 is the official SQLite extension distributed as the single C source file
`vec1.c`. It is unrelated to the third-party `sqlite-vec` package; do not install
`sqlite-vec`, because it uses a different schema and SQL API.

The project uses the `work` pyenv environment. On Ubuntu/WSL x86-64:

```bash
cd /home/morioka/embedding-explorer
pyenv activate work
python -m pip install -r requirements.txt
./scripts/build_vec1.sh
export VEC1_EXTENSION="$PWD/build/vec1.so"
```

The build script downloads the current official source from
[sqlite.org/vec1](https://sqlite.org/vec1/doc/trunk/doc/vec1.md), compiles it with
AVX2/FMA, links it to the SQLite runtime bundled by `pysqlite3-binary`, and verifies
`SELECT vec1_info()`. The expected result includes `Vec1 0.7`.

Start the application in the same shell:

```bash
pyenv activate work
export VEC1_EXTENSION="$PWD/build/vec1.so"
streamlit run app.py
```

`explorer.store` also searches `build/vec1.so`, so the environment variable may be
omitted when running from the project root. It is useful to set it explicitly in
scripts and services.

The default compile flags require an x86-64 CPU with AVX2 and FMA. On unsupported
hardware, edit `scripts/build_vec1.sh` and remove `-mavx2 -mfma`; the source has a
portable scalar path. Python's SQLite implementation must support extension loading;
the included `pysqlite3-binary` dependency provides that support in `work`.

Search uses the official SQL API:

```sql
CREATE VIRTUAL TABLE sample_vectors USING vec1(vector);
INSERT INTO sample_vectors(rowid, vector) VALUES(123, :float32_blob);
INSERT INTO sample_vectors(cmd, arg)
  VALUES('rebuild', '{index:"flat", distance:"cos"}');
SELECT rowid, vec1_cos_distance(:query_blob, vector) AS distance
  FROM sample_vectors(:query_blob, '{k:10}') ORDER BY distance;
```

The default is **exact flat NN** and not ANN, appropriate to exploratory datasets
below roughly 5,000 embeddings. ANN requires explicit `vec1_train()` and an index
`rebuild` with the trained model, plus candidate reranking. This package does not
silently enable ANN. Vec1 is not `sqlite-vec` and existing vec0 databases cannot
be reused directly; re-import the NPZ into a fresh Vec1 database.
