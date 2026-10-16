"""Persist training coordinates in SQLite and the fitted mapper in a local file.

Only load projector pickle files created by this app in a trusted directory.
"""
import hashlib
import json
import os
import pickle
from pathlib import Path
from .core import fit_projection
from .store import load_projection, save_projection


def key_for(revision, method, neighbors, seed=42):
    settings = {'revision': revision, 'method': method, 'neighbors': neighbors if method == 'UMAP' else None,
                'seed': seed, 'schema': 1}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]


def get_or_fit(conn, data_dir, vectors, ids, revision, method, neighbors, seed=42):
    key = key_for(revision, method, neighbors, seed)
    cache_dir = Path(data_dir) / 'projection_models'
    model_file = cache_dir / f'{key}.pkl'
    xy = load_projection(conn, key, ids)
    if xy is not None and model_file.is_file():
        with model_file.open('rb') as f:
            return pickle.load(f), xy, False

    projector, xy = fit_projection(vectors, method, neighbors, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f'{key}.{os.getpid()}.tmp'
    try:
        with tmp.open('wb') as f:
            pickle.dump(projector, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, model_file)
        save_projection(conn, key, ids, xy)
    finally:
        tmp.unlink(missing_ok=True)
    return projector, xy, True
