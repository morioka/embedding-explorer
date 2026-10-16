import numpy as np
import pytest
from explorer.core import fit_projection, transform_query, numpy_knn
from explorer.dataset import save_dataset, load_dataset


def test_pca_and_query():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 8))
    model, xy = fit_projection(x, "PCA")
    assert xy.shape == (20, 2)
    assert transform_query(model, x[3]).shape == (2,)
    assert numpy_knn(x, x[3], 1)[0][0] == 3


def test_umap_transform():
    pytest.importorskip("umap")
    rng = np.random.default_rng(9)
    x = rng.normal(size=(20, 8))
    model, xy = fit_projection(x, "UMAP", neighbors=5)
    assert xy.shape == (20, 2)
    assert np.isfinite(transform_query(model, x[2])).all()


def test_round_trip(tmp_path):
    x = np.eye(3, dtype=np.float32)
    save_dataset(tmp_path, x, ["a", "b", "c"], ["a.png", "b.png", "c.png"])
    output, labels, paths, metadata = load_dataset(tmp_path)
    assert output.shape == (3, 3)
    assert labels == ["a", "b", "c"]


def test_sqlite_vec1(tmp_path):
    import os
    from pathlib import Path
    if not os.environ.get('VEC1_EXTENSION') or not Path(os.environ['VEC1_EXTENSION']).is_file():
        pytest.skip('official Vec1 binary not configured')
    from explorer.store import connect, initialize, search
    x = np.eye(5, dtype=np.float32)
    conn = connect(tmp_path / "test.db")
    initialize(conn, x, list("abcde"), [f"{i}.png" for i in range(5)])
    output = search(conn, x[3], 3)
    assert output[0][0] == 3
    assert abs(output[0][1]) < 1e-5
    conn.close()
