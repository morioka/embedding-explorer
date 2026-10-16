from pathlib import Path
import numpy as np
from PIL import Image
from .core import unit_vectors


def _ensure_torch_hf_storage_compat():
    """Bridge torch 2.7's private HF checkpoint module to Transformers 5.x."""
    import importlib
    import sys
    import types

    module_name = "torch.distributed.checkpoint.hf_storage"
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        private = importlib.import_module("torch.distributed.checkpoint._hf_storage")
    compat = types.ModuleType(module_name)
    compat.HuggingFaceStorageReader = private._HuggingFaceStorageReader
    compat.HuggingFaceStorageWriter = private._HuggingFaceStorageWriter
    sys.modules[module_name] = compat


class EmbeddingGemma2:
    def __init__(self, model_path):
        _ensure_torch_hf_storage_compat()
        from sentence_transformers import SentenceTransformer
        p = Path(model_path).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"Local model directory is missing: {p}. Download separately; remote auto-download disabled.")
        self.model = SentenceTransformer(str(p), local_files_only=True)

    def image(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        with image.convert("RGB") as rgb:
            raw = self.model.encode({"image": rgb.copy()}, convert_to_numpy=True)
        return unit_vectors(raw)[0]

    def text(self, text):
        raw = self.model.encode(text, prompt_name="Retrieval-query", convert_to_numpy=True)
        return unit_vectors(raw)[0]
