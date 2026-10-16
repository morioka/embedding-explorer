"""Build genuine EmbeddingGemma 2 image vectors from data/images/ (local model only)."""
import argparse
from pathlib import Path
import sys
from explorer.dataset import save_dataset
from explorer.embedder import EmbeddingGemma2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Full LOCAL model directory")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    root = Path(args.data).resolve()
    paths = sorted(p for p in (root / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if len(paths) < 3:
        sys.exit("Need >=3 image files in data/images")
    model = EmbeddingGemma2(args.model)
    vectors, labels, relative_paths = [], [], []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}", flush=True)
        vectors.append(model.image(path))
        labels.append(path.stem)
        relative_paths.append(str(path.relative_to(root)))
    save_dataset(root, vectors, labels, relative_paths, model_id=str(Path(args.model).resolve()))
    print("Saved embeddings.npz and manifest.json")

if __name__ == "__main__":
    main()
