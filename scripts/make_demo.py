"""Create a deterministic synthetic image/vector dataset, no model download required."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from explorer.dataset import save_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    images = root / "data" / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    dim, n_each = 48, 18
    vectors, labels, paths = [], [], []
    colors = [(227,105,91),(70,160,211),(82,178,137),(176,129,211)]
    for group, color in enumerate(colors):
        center = rng.normal(size=dim)
        center /= np.linalg.norm(center)
        for j in range(n_each):
            vec = center + rng.normal(scale=0.18, size=dim)
            vectors.append(vec)
            path = images / f"demo_{group}_{j:02d}.png"
            im = Image.new("RGB", (144, 144), (248, 248, 248))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((14, 14, 130, 130), radius=22, fill=color)
            variation = int(rng.integers(14, 70))
            if group % 2:
                d.ellipse((variation//2, 35, variation+45, 95), fill="white")
            else:
                d.rectangle((variation//2, 38, variation+44, 100), fill="white")
            im.save(path)
            labels.append(f"group-{group+1} / sample-{j+1:02d}")
            paths.append(str(path.relative_to(root / "data")))
    save_dataset(root / "data", np.asarray(vectors), labels, paths, model_id="synthetic-demo")
    print("Created 72 synthetic samples in data/ (not real image embeddings)")

if __name__ == "__main__":
    main()
