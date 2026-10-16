from pathlib import Path
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
from explorer.dataset import load_dataset
from explorer.core import fit_projection, transform_query, numpy_knn, unit_vectors
from explorer.store import connect, initialize, search, has_dataset, load, get_metadata
from explorer.projection_cache import get_or_fit
from explorer.ingest import prepare_uploads, append_uploads

st.set_page_config(page_title="Embedding Explorer", layout="wide")
st.title("Embedding Explorer")
st.caption("2D projection is visualization only; nearest neighbors are computed in the original vector space.")

@st.cache_resource(show_spinner="Loading local EmbeddingGemma 2...")
def get_embedder(path):
    from explorer.embedder import EmbeddingGemma2
    return EmbeddingGemma2(path)


def open_image(path):
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (OSError, ValueError):
        return None


def draw_figure(xy, labels, paths, root, query_xy, neighbors, thumbnail, max_thumbnails):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=9, color="#6798cd", opacity=0.7), customdata=np.arange(len(xy))[:, None], text=labels, hovertemplate="%{text}<extra></extra>", name="Samples"))
    if neighbors:
        indices = [index for index, _ in neighbors]
        fig.add_trace(go.Scatter(x=xy[indices, 0], y=xy[indices, 1], mode="markers+text", text=[str(i+1) for i in range(len(indices))], textposition="top center", marker=dict(size=17, color="#eea54d", line=dict(color="white", width=2)), customdata=np.asarray(indices)[:, None], hovertext=[labels[i] for i in indices], hovertemplate="%{hovertext}<extra></extra>", name="Nearest"))
    if query_xy is not None:
        fig.add_trace(go.Scatter(x=[float(query_xy[0])], y=[float(query_xy[1])], mode="markers", marker=dict(size=19, color="#e45152", symbol="star", line=dict(color="white", width=2)), name="Query", hovertemplate="Query<extra></extra>"))
    if thumbnail:
        indices = np.arange(len(xy))
        if len(indices) > max_thumbnails:
            indices = np.unique(np.linspace(0, len(xy)-1, max_thumbnails).astype(int))
        spread = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
        size = spread * 0.065
        for i in indices:
            im = open_image(root / paths[i])
            if im is None:
                continue
            im.thumbnail((64, 64))
            fig.add_layout_image(dict(source=im, x=float(xy[i, 0]), y=float(xy[i, 1]), sizex=size, sizey=size, xref="x", yref="y", xanchor="center", yanchor="middle", layer="above"))
    fig.update_layout(height=630, margin=dict(t=15, b=15, l=15, r=15), legend=dict(orientation="h"), dragmode="pan", xaxis_title="dimension 1", yaxis_title="dimension 2")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

with st.sidebar:
    st.header("Settings")
    data_dir = Path(st.text_input("Data directory", "data")).expanduser().resolve()
    method = st.radio("Projection", ["UMAP", "PCA"], horizontal=True)
    neighbors = st.slider("UMAP n_neighbors", 2, 100, 15, disabled=method != "UMAP")
    k = st.slider("k nearest", 1, 30, 5)
    thumb = st.checkbox("Thumbnails on map", value=True)
    max_thumb = st.slider("Max thumbnails", 10, 200, 80, disabled=not thumb)
    backend = st.radio("Nearest neighbor backend", ["numpy", "SQLite Vec1"], horizontal=True)
    st.divider()
    st.subheader("Optional embedding model")
    model_path = st.text_input("Local model directory", os.environ.get("EMBEDDING_MODEL_PATH", ""), help="The entire downloaded HF model directory; no automatic remote download")

# Import new images from Streamlit: embeddings -> SQLite -> refit UMAP/PCA -> redraw.
with st.expander("Add sample images (batch)", expanded=True):
    st.caption("Drop multiple PNG/JPEG/WebP images. After clicking Import, embeddings are generated, duplicates skipped, and the projection is recomputed automatically.")
    new_images = st.file_uploader("Drop sample images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="batch_samples")
    if st.button("Embed, store and reproject", disabled=not new_images, type="primary"):
        try:
            if not model_path.strip():
                raise ValueError("Set Local model directory in the sidebar first.")
            prepared, errors = prepare_uploads(new_images)
            for message in errors:
                st.warning(message)
            if not prepared:
                raise ValueError("No valid images were supplied.")
            with st.spinner(f"Embedding {len(prepared)} images and storing vectors..."):
                ingest_conn = connect(data_dir / "vectors.db")
                # Do not silently blend new model vectors into the demo or existing unrelated embeddings.
                if not has_dataset(ingest_conn) and (data_dir / "embeddings.npz").is_file():
                    import_vectors, import_labels, import_paths, import_manifest = load_dataset(data_dir)
                    initialize(ingest_conn, import_vectors, import_labels, import_paths, import_manifest.get("model_id", "external"))
                model_id = str(Path(model_path).expanduser().resolve())
                added, skipped = append_uploads(ingest_conn, data_dir, prepared, get_embedder(model_path), model_id)
                if added:
                    all_vectors, _, _, meta, all_ids = load(ingest_conn)
                    if len(all_vectors) >= 3:
                        get_or_fit(ingest_conn, data_dir, all_vectors, all_ids, meta["revision"], method, neighbors)
                ingest_conn.close()
            st.session_state["ingest_notice"] = f"Imported {added} images; skipped {skipped} already indexed. Projection updated for {method}." if added else f"No new images: {skipped} duplicates."
            st.rerun()
        except Exception as exc:
            st.error(f"Image import failed: {exc}")

if st.session_state.get("ingest_notice"):
    st.success(st.session_state.pop("ingest_notice"))

# SQLite is authoritative. A legacy NPZ is imported only when the DB is empty.
try:
    conn = connect(data_dir / 'vectors.db')
    if not has_dataset(conn):
        if not (data_dir / 'embeddings.npz').is_file():
            st.info('No samples in SQLite. Run scripts/make_demo.py, scripts/build_images.py, or provide data/embeddings.npz.')
            st.stop()
        import_vectors, import_labels, import_paths, import_manifest = load_dataset(data_dir)
        initialize(conn, import_vectors, import_labels, import_paths, import_manifest.get('model_id', 'external'))
        st.toast('Imported existing NPZ into SQLite')
    vectors, labels, paths, manifest, sample_ids = load(conn)
    if len(vectors) < 3:
        st.info('At least 3 samples are needed for the 2D projection. Add more images above.')
        st.stop()
    projector, xy, refitted = get_or_fit(conn, data_dir, vectors, sample_ids,
        manifest['revision'], method, neighbors)
except Exception as exc:
    st.error(f'Unable to open data or projection: {exc}')
    st.stop()

st.caption(f"{len(vectors)} samples | {vectors.shape[1]} dimensions | origin: {manifest.get('model_id','unknown')} | revision: {manifest['revision']}")
if refitted:
    st.info('Projection trained and saved to SQLite + projection_models/. Subsequent reruns reuse it.')

col_left, col_right = st.columns([3, 2])
with col_right:
    st.subheader("Query")
    mode = st.radio("Query source", ["Existing sample", "Upload vector", "Upload image", "Text"], horizontal=False)
    query = None
    if mode == "Existing sample":
        index = st.selectbox("Sample", range(len(labels)), format_func=lambda idx: f"{idx+1}: {labels[idx]}")
        query = vectors[index]
    elif mode == "Upload vector":
        uploaded = st.file_uploader("Vector: .npy or .npz (key: vector)", type=["npy", "npz"])
        if uploaded:
            try:
                with np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=False) if uploaded.name.endswith(".npz") else io.BytesIO(uploaded.getvalue()) as stream:
                    raw = stream["vector"] if uploaded.name.endswith(".npz") else np.load(stream, allow_pickle=False)
                query = unit_vectors(raw)[0]
            except Exception as exc:
                st.error(f"Invalid vector file: {exc}")
    elif mode == "Upload image":
        uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded:
            image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            st.image(image, width=180)
            if st.button("Embed image"):
                try:
                    query = get_embedder(model_path).image(image)
                    st.session_state.query_image = query
                    st.session_state.query_image_signature = uploaded.file_id if hasattr(uploaded, "file_id") else uploaded.name + str(len(uploaded.getvalue()))
                except Exception as exc:
                    st.error(str(exc))
            key = uploaded.file_id if hasattr(uploaded, "file_id") else uploaded.name + str(len(uploaded.getvalue()))
            if st.session_state.get("query_image_signature") == key:
                query = st.session_state.query_image
    else:
        text_query = st.text_input("Text query")
        if st.button("Embed text") and text_query:
            try:
                query = get_embedder(model_path).text(text_query)
                st.session_state.query_text_vector = query
                st.session_state.query_text_source = text_query
            except Exception as exc:
                st.error(str(exc))
        if text_query and st.session_state.get("query_text_source") == text_query:
            query = st.session_state.query_text_vector

    results = []
    projected = None
    if query is not None:
        if len(query) != vectors.shape[1]:
            st.error(f"Dimension mismatch: query={len(query)}, dataset={vectors.shape[1]}")
        elif manifest.get("model_id") == "synthetic-demo" and mode in ("Upload image", "Text"):
            st.error("Demo vectors are synthetic. Use a dataset generated by the same embedding model as your query.")
        else:
            projected = transform_query(projector, query)
            results = numpy_knn(vectors, query, k) if backend == "numpy" else search(conn, query, k, sample_ids)
            st.subheader("Nearest samples")
            st.caption("Distance: cosine distance (smaller is closer); score = 1 − distance")
            for rank, (idx, distance) in enumerate(results, 1):
                a, b = st.columns([1, 3])
                with a:
                    image = open_image(data_dir / paths[idx])
                    if image is not None:
                        st.image(image, width=95)
                with b:
                    st.markdown(f"**#{rank} {labels[idx]}**")
                    st.caption(f"id={idx+1} · distance={distance:.5f} · similarity={1-distance:.5f}")

with col_left:
    st.subheader("2D projection")
    figure = draw_figure(xy, labels, paths, data_dir, projected, results, thumb, max_thumb)
    event = st.plotly_chart(figure, width="stretch", on_select="rerun", selection_mode=["points", "box"], key=f"map_{method}_{manifest['revision']}_{neighbors}_{thumb}_{max_thumb}")
    chosen = event.selection.points if event is not None else []
    selected_indices = []
    for item in chosen:
        c = item.get("customdata")
        if c is not None and len(c):
            selected_indices.append(int(c[0]))
    selected_indices = sorted(set(selected_indices))
    if selected_indices:
        st.write(f"Selected: {len(selected_indices)}")
        for idx in selected_indices[:12]:
            st.write(f"{idx+1}: {labels[idx]}")
            image = open_image(data_dir / paths[idx])
            if image is not None:
                st.image(image, width=130)

with st.expander("Dataset format"):
    st.code('np.savez_compressed("data/embeddings.npz", vectors=X, labels=labels, paths=paths)')
    st.write("X: (N,D) float array. paths: relative to the selected data directory. Optional manifest.json sets model_id.")
