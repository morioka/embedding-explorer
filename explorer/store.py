"""SQLite is authoritative; official SQLite Vec1 provides cosine nearest-neighbor search."""
import io
import os
import json
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3
from pathlib import Path
import numpy as np
from .core import unit_vectors


def connect(path):
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # SQLite's official Vec1 is a locally compiled loadable extension.
    # Prefer explicit VEC1_EXTENSION, then a sibling of the DB, then project build/.
    candidate_paths = [os.environ.get('VEC1_EXTENSION'),
                       str(Path(path).parent / 'vec1.so'),
                       str(Path(__file__).resolve().parent.parent / 'build' / 'vec1.so')]
    extension = next((item for item in candidate_paths if item and Path(item).is_file()), None)
    if extension is None:
        conn.close()
        raise RuntimeError('Vec1 extension not found. Compile vec1.c, then set VEC1_EXTENSION=/absolute/path/vec1.so')
    try:
        conn.enable_load_extension(True)
        conn.load_extension(extension)
    finally:
        conn.enable_load_extension(False)
    conn.execute('SELECT vec1_info()').fetchone()
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, label TEXT NOT NULL, image_path TEXT NOT NULL, embedding BLOB NOT NULL, sha256 TEXT)')
    if 'sha256' not in [r[1] for r in conn.execute('PRAGMA table_info(samples)')]:
        conn.execute('ALTER TABLE samples ADD COLUMN sha256 TEXT')
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS samples_sha256_idx ON samples(sha256)')
    conn.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    conn.execute('''CREATE TABLE IF NOT EXISTS projections (
        projection_key TEXT NOT NULL, sample_id INTEGER NOT NULL,
        x REAL NOT NULL, y REAL NOT NULL,
        PRIMARY KEY(projection_key, sample_id),
        FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE)''')
    conn.commit()
    return conn


def has_dataset(conn):
    return conn.execute('SELECT count(*) FROM samples').fetchone()[0] > 0


def get_metadata(conn):
    return {k: json.loads(v) for k, v in conn.execute('SELECT key,value FROM metadata')}


def initialize(conn, vectors, labels, paths, model_id='external'):
    x = unit_vectors(vectors)
    n, dim = x.shape
    if len(labels) != n or len(paths) != n:
        raise ValueError('Metadata count differs from vector count')
    with conn:
        conn.execute('DROP TABLE IF EXISTS sample_vectors')
        conn.execute('DELETE FROM projections')
        conn.execute('DELETE FROM samples')
        conn.execute('CREATE VIRTUAL TABLE sample_vectors USING vec1(vector)')
        conn.executemany('INSERT INTO samples(id,label,image_path,embedding) VALUES(?,?,?,?)',
            [(i+1, str(labels[i]), str(paths[i]), x[i].tobytes()) for i in range(n)])
        conn.executemany('INSERT INTO sample_vectors(rowid,vector) VALUES(?,?)',
            [(i+1, x[i].astype('<f4', copy=False).tobytes()) for i in range(n)])
        conn.execute("INSERT INTO sample_vectors(cmd,arg) VALUES('rebuild', ?)",
                     (json.dumps({'index':'flat', 'distance':'cos'}),))
        revision = int(get_metadata(conn).get('revision', 0)) + 1
        for key, val in {'revision': revision, 'model_id': model_id, 'dimension': dim, 'count': n}.items():
            conn.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)', (key, json.dumps(val)))
    return revision


def load(conn):
    rows = conn.execute('SELECT id,label,image_path,embedding FROM samples ORDER BY id').fetchall()
    if not rows:
        raise ValueError('Database has no samples')
    vectors = np.stack([np.frombuffer(r[3], dtype=np.float32).copy() for r in rows])
    return vectors, [r[1] for r in rows], [r[2] for r in rows], get_metadata(conn), [r[0] for r in rows]


def save_projection(conn, key, ids, xy):
    if len(ids) != len(xy):
        raise ValueError('Projection size does not match samples')
    with conn:
        conn.execute('DELETE FROM projections WHERE projection_key=?', (key,))
        conn.executemany('INSERT INTO projections(projection_key,sample_id,x,y) VALUES(?,?,?,?)',
                         [(key, int(sample_id), float(pos[0]), float(pos[1])) for sample_id,pos in zip(ids, xy)])


def load_projection(conn, key, ids):
    coords = dict((sample_id, (x,y)) for sample_id,x,y in conn.execute(
        'SELECT sample_id,x,y FROM projections WHERE projection_key=?', (key,)))
    if len(coords) != len(ids) or any(i not in coords for i in ids):
        return None
    return np.asarray([coords[i] for i in ids], dtype=np.float32)


def search(conn, query, k, ids=None):
    q = unit_vectors(query)[0]
    # Vec1's table-valued function returns nearest candidates, but not a distance column.
    # Compute exact cosine distances on the returned original vectors.
    qblob = q.astype('<f4', copy=False).tobytes()
    count = conn.execute('SELECT count(*) FROM samples').fetchone()[0]
    k = min(max(int(k), 1), count)
    rows = conn.execute('''SELECT rowid, vec1_cos_distance(?, vector) AS distance
        FROM sample_vectors(?, ?) ORDER BY distance''',
        (qblob, qblob, json.dumps({'k': k}))).fetchall()
    index = {sample_id:i for i,sample_id in enumerate(ids)} if ids is not None else None
    return [(index[int(rowid)] if index is not None else int(rowid)-1, float(distance)) for rowid,distance in rows]
