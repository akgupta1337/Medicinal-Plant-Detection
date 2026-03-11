import random
import shutil
from pathlib import Path

# --- Config (change these values directly if you want) ---
RAW_DIR = Path("dataset/raw")
OUT_DIR = Path("dataset")
SPLIT = 0.15  # fraction used for val and test each
SEED = 42


def copy_split(src_dir: Path, dest_root: Path, split: float, seed: int = 42):
    """Split a folder of class subfolders into train/val/test."""

    random.seed(seed)
    classes = [d.name for d in src_dir.iterdir() if d.is_dir()]
    if not classes:
        raise FileNotFoundError(f"No class directories found in {src_dir}")

    for split_name in ["train", "val", "test"]:
        (dest_root / split_name).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        src_cls_dir = src_dir / cls
        images = [p for p in src_cls_dir.iterdir() if p.is_file()]
        random.shuffle(images)

        n = len(images)
        n_val = int(n * split)
        n_test = n_val
        n_train = n - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, items in splits.items():
            dest_dir = dest_root / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in items:
                shutil.copy2(item, dest_dir / item.name)

    print(f"Done splitting dataset to {dest_root}")


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {RAW_DIR}")

    copy_split(RAW_DIR, OUT_DIR, SPLIT, SEED)


if __name__ == "__main__":
    main()
