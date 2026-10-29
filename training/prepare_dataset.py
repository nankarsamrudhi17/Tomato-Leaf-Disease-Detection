"""
Build dataset/train, dataset/val, dataset/test from:
  1) PlantVillage tomato folders  (--pv)
  2) your own webcam images        (--real)   <- includes Not_a_leaf

Usage (run on your PC):
  python prepare_dataset.py --pv "C:/data/PlantVillage" --real "../data_collection/webcam_images" --out dataset

Then zip the 'dataset' folder as leaf_dataset.zip and upload it to Google Drive.

How the split avoids leakage
  * PlantVillage: images are grouped by filename (flipped/rotated copies of the same
    leaf share a group) and each GROUP goes entirely to train, val or test.
    This is best-effort; your real webcam test images are the true test.
  * Webcam images: sorted by filename (timestamp) and split in CONTIGUOUS blocks, so
    near-identical frames taken one after another do not end up in both train and test.
"""
import argparse
import hashlib
import random
import re
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15          # rest (0.15) = test

# PlantVillage folder name (lowercased, spaces -> _) contains key -> project class name
PV_MAP = [
    ("bacterial_spot",   "Tomato_Bacterial_spot"),
    ("early_blight",     "Tomato_Early_blight"),
    ("late_blight",      "Tomato_Late_blight"),
    ("leaf_mold",        "Tomato_Leaf_Mold"),
    ("septoria",         "Tomato_Septoria_leaf_spot"),
    ("spider_mites",     "Tomato_Spider_mites"),
    ("target_spot",      "Tomato_Target_Spot"),
    ("yellow_leaf_curl", "Tomato_Yellow_Leaf_Curl_Virus"),
    ("mosaic_virus",     "Tomato_Mosaic_virus"),
    ("healthy",          "Tomato_Healthy"),
]

AUG_SUFFIX = re.compile(
    r"(_flip(tb|lr)|_(90|180|270)deg|_rot\d*|_scaling|_noise\w*|_contrast|_brightness|_gamma|_shear)+$",
    re.IGNORECASE,
)


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def map_pv_class(folder_name: str):
    key = folder_name.lower().replace(" ", "_")
    if not key.startswith("tomato"):
        return None
    for frag, cls in PV_MAP:
        if frag in key:
            return cls
    return None


def group_split(path: Path) -> str:
    """Deterministic split by filename group (keeps augmented copies together)."""
    group = AUG_SUFFIX.sub("", path.stem.lower())
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 10000 / 10000
    if h < TRAIN_FRAC:
        return "train"
    if h < TRAIN_FRAC + VAL_FRAC:
        return "val"
    return "test"


def block_split(files):
    """Contiguous-block split for sequentially captured webcam frames."""
    n = len(files)
    a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return {"train": files[:a], "val": files[a:b], "test": files[b:]}


def copy_files(files, dst: Path, prefix: str):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f"{prefix}{f.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv", help="PlantVillage root folder (contains class subfolders)")
    ap.add_argument("--real", help="webcam_images folder (class subfolders)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--max-train", type=int, default=1500,
                    help="max PlantVillage training images per class (reduces imbalance)")
    ap.add_argument("--classes", nargs="*", help="only use these project classes (default: all found)")
    ap.add_argument("--overwrite", action="store_true", help="delete --out first")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        if args.overwrite:
            shutil.rmtree(out)
        else:
            raise SystemExit(f"{out} already exists. Use --overwrite to rebuild it.")
    wanted = set(args.classes) if args.classes else None
    rng = random.Random(42)
    summary = {}

    if args.pv:
        for d in sorted(Path(args.pv).iterdir()):
            if not d.is_dir():
                continue
            cls = map_pv_class(d.name)
            if cls is None or (wanted and cls not in wanted):
                continue
            splits = {"train": [], "val": [], "test": []}
            for f in list_images(d):
                splits[group_split(f)].append(f)
            if len(splits["train"]) > args.max_train:
                rng.shuffle(splits["train"])
                splits["train"] = splits["train"][: args.max_train]
            for s, files in splits.items():
                copy_files(files, out / s / cls, "pv_")
                summary.setdefault(cls, {"train": 0, "val": 0, "test": 0})[s] += len(files)

    if args.real:
        for d in sorted(Path(args.real).iterdir()):
            if not d.is_dir():
                continue
            cls = d.name
            if wanted and cls not in wanted:
                continue
            files = list_images(d)
            if len(files) < 20:
                print(f"WARNING: only {len(files)} webcam images for '{cls}' - collect more (50-100+).")
            for s, fl in block_split(files).items():
                copy_files(fl, out / s / cls, "real_")
                summary.setdefault(cls, {"train": 0, "val": 0, "test": 0})[s] += len(fl)

    if not summary:
        raise SystemExit("No images copied. Check --pv / --real paths.")

    print(f"\n{'class':35s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for cls in sorted(summary):
        c = summary[cls]
        print(f"{cls:35s} {c['train']:6d} {c['val']:6d} {c['test']:6d}")
        if c["val"] == 0 or c["test"] == 0:
            print(f"  ^ WARNING: '{cls}' has an empty val/test split")
    print(f"\nDone -> {out.resolve()}")
    print("Next: zip the folder as leaf_dataset.zip (top-level folder must be 'dataset') and upload to Google Drive.")
    print("Note: class folders must be identical in train/val/test or the notebook will stop.")


if __name__ == "__main__":
    main()
