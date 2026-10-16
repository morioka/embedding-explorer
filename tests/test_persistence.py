import numpy as np
from PIL import Image
from explorer.projection_cache import get_or_fit
from explorer.store import connect, initialize, load, load_projection, search


def test_sqlite_source_and_cached_projection(tmp_path):
    try:
        import os
        from pathlib import Path
        if not (os.environ.get('VEC1_EXTENSION') and Path(os.environ['VEC1_EXTENSION']).is_file()):
            import pytest
            pytest.skip('official Vec1 binary not configured')
    except ImportError:
        import pytest
        pytest.skip('official Vec1 compiled binary not available')
    db = connect(tmp_path / 'vectors.db')
    vectors = np.random.default_rng(7).normal(size=(15, 8)).astype('float32')
    revision = initialize(db, vectors, [f's{i}' for i in range(15)], [f'{i}.jpg' for i in range(15)])
    x, labels, paths, meta, ids = load(db)
    assert x.shape == vectors.shape
    assert meta['revision'] == revision
    projector, xy, refitted = get_or_fit(db, tmp_path, x, ids, revision, 'PCA', 15)
    assert refitted and xy.shape == (15,2)
    projector2, xy2, refitted2 = get_or_fit(db, tmp_path, x, ids, revision, 'PCA', 15)
    assert not refitted2 and np.allclose(xy, xy2)
    assert search(db, x[0], 1, ids)[0][0] == 0


def test_vec1_rebuild_allows_multiple_upload_batches(tmp_path):
    import hashlib
    import os
    import pytest
    from pathlib import Path
    from explorer.ingest import append_uploads

    if not (os.environ.get('VEC1_EXTENSION') and Path(os.environ['VEC1_EXTENSION']).is_file()):
        pytest.skip('official Vec1 binary not configured')
    db = connect(tmp_path / 'vectors.db')
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(3, 8)).astype('float32')
    initialize(db, vectors, ['a', 'b', 'c'], ['a.jpg', 'b.jpg', 'c.jpg'], '/model')

    class Embedder:
        def image(self, _image):
            return np.ones(8, dtype='float32')

    image = Image.new('RGB', (4, 4), 'red')
    for payload in (b'first', b'second'):
        digest = hashlib.sha256(payload).hexdigest()
        added, skipped = append_uploads(
            db, tmp_path, [(f'{digest}.png', payload, digest, image)], Embedder(), '/model')
        assert (added, skipped) == (1, 0)
    assert db.execute('SELECT count(*) FROM samples').fetchone()[0] == 5
