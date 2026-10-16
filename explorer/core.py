import numpy as np
from sklearn.decomposition import PCA


def unit_vectors(vectors):
    a = np.asarray(vectors, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Vectors must be a finite 2D numeric array")
    lengths = np.linalg.norm(a, axis=1, keepdims=True)
    if (lengths < 1e-12).any():
        raise ValueError("Zero vectors are not supported")
    return np.ascontiguousarray(a / lengths)


def fit_projection(vectors, method="UMAP", neighbors=15, seed=42):
    x = unit_vectors(vectors)
    if len(x) < 3:
        raise ValueError("At least 3 samples are required")
    if method == "PCA":
        projector = PCA(n_components=2, random_state=seed)
    elif method == "UMAP":
        import umap
        projector = umap.UMAP(n_components=2, metric="cosine", n_neighbors=max(2, min(neighbors, len(x)-1)), random_state=seed, transform_seed=seed)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    xy = projector.fit_transform(x)
    return projector, np.asarray(xy, dtype=np.float32)


def transform_query(projector, vector):
    return np.asarray(projector.transform(unit_vectors(vector)), dtype=np.float32)[0]


def numpy_knn(vectors, query, k):
    x = unit_vectors(vectors)
    q = unit_vectors(query)[0]
    similarity = x @ q
    order = np.argsort(-similarity, kind="stable")[:min(k, len(x))]
    return [(int(i), float(1 - similarity[i])) for i in order]
