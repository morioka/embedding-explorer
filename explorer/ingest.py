"""Batch ingestion with content-hash deduplication and dataset revision tracking."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .core import unit_vectors
from .store import get_metadata

ALLOWED_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def prepare_uploads(files, max_bytes=20 * 1024 * 1024):
    """Validate and decode image uploads, removing duplicates within a batch."""
    prepared, errors, seen = [], [], set()
    for uploaded in files:
        name = Path(uploaded.name).name
        if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            errors.append(f'{name}: unsupported extension')
            continue
        data = uploaded.getvalue()
        if len(data) > max_bytes:
            errors.append(f'{name}: over size limit')
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        try:
            with Image.open(io.BytesIO(data)) as src:
                src.load()
                rgb = ImageOps.exif_transpose(src).convert('RGB')
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            errors.append(f'{name}: invalid image: {exc}')
            continue
        seen.add(digest)
        prepared.append((name, data, digest, rgb))
    return prepared, errors


def append_uploads(conn, data_dir, prepared, embedder, model_id):
    """Embed first, then atomically append SQLite rows and official Vec1 entries.

    Files are written after embeddings but before the DB transaction. On error,
    newly created files are rolled back. Pre-existing files remain untouched.
    """
    if not prepared:
        return 0, 0
    meta = get_metadata(conn)
    previous = conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
    if previous and meta.get('model_id') != model_id:
        raise ValueError(f"Model mismatch: dataset={meta.get('model_id')!r}; selected={model_id!r}. Use a separate data directory for this model.")
    unique = []
    for name, payload, digest, rgb in prepared:
        existing = conn.execute('SELECT id FROM samples WHERE sha256=?', (digest,)).fetchone()
        if existing is None:
            unique.append((name, payload, digest, rgb))
    if not unique:
        return 0, len(prepared)
    vectors = []
    for name, payload, digest, rgb in unique:
        vectors.append(embedder.image(rgb))
    x = unit_vectors(np.asarray(vectors, dtype=np.float32))
    dim = int(x.shape[1])
    if previous and dim != int(meta['dimension']):
        raise ValueError(f"Embedding dimension mismatch: database={meta['dimension']}, new={dim}")
    images_dir = Path(data_dir) / 'images' / 'uploaded'
    images_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for name, payload, digest, rgb in unique:
            filepath = images_dir / (digest + Path(name).suffix.lower())
            if not filepath.exists():
                filepath.write_bytes(payload)
                created.append(filepath)
        with conn:
            start = int(conn.execute('SELECT COALESCE(MAX(id),0) FROM samples').fetchone()[0])
            for i, (name, payload, digest, rgb) in enumerate(unique):
                row_id = start + i + 1
                relpath = str((images_dir / (digest + Path(name).suffix.lower())).relative_to(data_dir))
                conn.execute('INSERT INTO samples(id,label,image_path,embedding,sha256) VALUES(?,?,?,?,?)',
                             (row_id, name, relpath, x[i].tobytes(), digest))
            # Vec1's rebuild compacts the base table, so appending directly to
            # an already rebuilt table fails with "unexpected vector size".
            # Recreate the index from SQLite-authoritative sample embeddings.
            conn.execute('DROP TABLE IF EXISTS sample_vectors')
            conn.execute('CREATE VIRTUAL TABLE sample_vectors USING vec1(vector)')
            conn.executemany('INSERT INTO sample_vectors(rowid,vector) VALUES(?,?)',
                             conn.execute('SELECT id,embedding FROM samples ORDER BY id'))
            conn.execute("INSERT INTO sample_vectors(cmd,arg) VALUES('rebuild', ?)",
                         (json.dumps({'index':'flat', 'distance':'cos'}),))
            revision = int(meta.get('revision', 0)) + 1
            for key, val in {'revision': revision, 'model_id': model_id, 'dimension': dim, 'count': previous + len(unique)}.items():
                conn.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)', (key, json.dumps(val)))
        return len(unique), len(prepared) - len(unique)
    except Exception:
        for file in created:
            file.unlink(missing_ok=True)
        raise
