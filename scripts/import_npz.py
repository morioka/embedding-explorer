"""Import a prepared NPZ into SQLite. Explicit --replace prevents accidental overwrites."""
import argparse
from pathlib import Path
from explorer.dataset import load_dataset
from explorer.store import connect, has_dataset, initialize

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
p.add_argument('--replace', action='store_true', help='Replace existing SQLite samples and invalidate old projections')
a = p.parse_args()
folder = Path(a.data)
conn = connect(folder / 'vectors.db')
if has_dataset(conn) and not a.replace:
    p.error('SQLite already contains samples; specify --replace to replace them')
vectors, labels, paths, manifest = load_dataset(folder)
revision = initialize(conn, vectors, labels, paths, manifest.get('model_id','external'))
print(f'Imported {len(vectors)} vectors (revision {revision}) into {folder / "vectors.db"}')
