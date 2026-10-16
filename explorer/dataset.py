from pathlib import Path
import json
import numpy as np
from .core import unit_vectors


def save_dataset(folder, vectors, labels, paths, model_id="external"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vectors = unit_vectors(vectors)
    if len(labels) != len(vectors) or len(paths) != len(vectors):
        raise ValueError("Metadata count differs from vector count")
    np.savez_compressed(folder / "embeddings.npz", vectors=vectors, labels=np.asarray(labels, dtype=str), paths=np.asarray(paths, dtype=str))
    (folder / "manifest.json").write_text(json.dumps({"model_id": model_id, "dimension": int(vectors.shape[1]), "count": int(len(vectors))}, indent=2), encoding="utf-8")


def load_dataset(folder):
    folder = Path(folder)
    with np.load(folder / "embeddings.npz", allow_pickle=False) as data:
        vectors = unit_vectors(data["vectors"])
        labels = data["labels"].astype(str).tolist()
        paths = data["paths"].astype(str).tolist()
    manifest_file = folder / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {"model_id": "unknown"}
    return vectors, labels, paths, manifest
