"""
data_process.py
===============
Prepares the European Art (DEArt) dataset for YOLOv8 training.

The script is split into two independent stages that can be run
together or individually via --stage:

  build     — Convert the DEArt dataset from COCO format to the YOLO
               flat-file layout expected by Ultralytics:
                 dataset_yolo/
                   images/{train,val}/
                   labels/{train,val}/
                   dataset.yaml

  augment   — Download the 5 k-image COCO val2017 split and merge it
               into dataset_yolo/ to improve generalisation on
               real-world objects that also appear in paintings.

Run all stages in sequence (default):
    python data_process.py

Run a single stage:
    python data_process.py --stage build --base-dir dataset_yolo --sample 42

Dataset
-------
    biglam/european_art  (HuggingFace Hub) — DEArt annotations in COCO JSON format.
    The dataset is downloaded and cached automatically on first run.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from ultralytics.utils.downloads import download

# ---------------------------------------------------------------------------
# Class index
# ---------------------------------------------------------------------------

# The 80 original COCO classes, in their canonical order (indices 0–79).
# These indices MUST remain fixed so fine-tuned YOLO weights stay compatible
# with any downstream tool that uses the standard COCO mapping.
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# DEArt classes that do NOT exist in COCO (indices 80–133).
# Classes that ARE in COCO (person, boat, bird, book, dog, horse, cow,
# sheep, elephant, cat, bear, zebra, apple, banana, orange, mouse) are
# deliberately omitted here — they reuse their COCO index directly.
DEART_NEW: list[str] = [
    "tree", "nude", "halo", "angel", "helmet", "lance", "knight",
    "sword", "jug", "banner", "crown", "prayer", "monk", "devil",
    "shield", "scroll", "chalice", "crucifixion", "donkey", "skull",
    "lion", "butterfly", "monkey", "lily", "serpent", "arrow", "palm",
    "dove", "trumpet", "key of heaven", "dragon", "mitre", "crozier",
    "tiara", "deer", "crown of thorns", "hands", "god the father",
    "eagle", "shepherd", "head", "camauro", "centaur", "swan", "rooster",
    "saturno", "unicorn", "zucchetto", "fish", "horn", "stole",
    "pegasus", "holy shroud", "judith",
]

# Merged class list: 80 COCO + 54 DEArt-exclusive = 134 total
ALL_CLASSES: list[str] = COCO_CLASSES + DEART_NEW

# name → integer index lookup used when writing YOLO label files
COCO_NAME2IDX: dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}
NAME2IDX: dict[str, int] = {
    **COCO_NAME2IDX,
    **{name: 80 + i for i, name in enumerate(DEART_NEW)},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_vivid_color(label: str) -> tuple[int, int, int]:
    """
    Returns a deterministic, visually distinct BGR colour for a class label.

    Seeded by the label string so the same class always gets the same colour
    across calls.  Uses high-saturation, high-value HSV to avoid dark or
    washed-out results.

    Args:
        label: Class name (e.g. "angel", "person").

    Returns:
        (B, G, R) tuple with each channel in [0, 255].
    """
    random.seed(label)
    h = random.randint(0, 179)
    s = random.randint(180, 255)
    v = random.randint(200, 255)

    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)


def draw_box_with_label(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """
    Draws a coloured bounding box and a filled label chip onto *img* in-place.

    The label chip is placed above the box when there is room, or inside the
    top edge when the box is too close to the image border.  Text colour
    (black / white) is chosen automatically for contrast.

    Args:
        img        : BGR image array (modified in-place).
        x1, y1     : Top-left corner of the bounding box.
        x2, y2     : Bottom-right corner of the bounding box.
        label      : Text to display.
        color      : BGR colour for both the box border and label chip background.
        font_scale : OpenCV font scale factor.
        thickness  : Line thickness for the box border and text stroke.
    """
    # Bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness + 1)

    # Measure the text chip dimensions
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # Clamp label position so it never overflows the top of the image
    y_text = max(y1 - 10, th + 10)

    # Filled background chip
    cv2.rectangle(
        img,
        (x1, y_text - th - baseline),
        (x1 + tw, y_text),
        color,
        -1,
    )

    # Black text on light backgrounds, white on dark
    text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)

    cv2.putText(
        img, label,
        (x1, y_text - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        text_color, thickness, cv2.LINE_AA,
    )


def coco_to_yolo(
    x: float, y: float, w: float, h: float, W: int, H: int
) -> tuple[float, float, float, float]:
    """
    Converts a COCO bounding box to the normalised YOLO format.

    COCO  : [x_top_left, y_top_left, width, height]  (pixels, absolute)
    YOLO  : [cx, cy, w, h]                           (normalised to [0, 1])

    Args:
        x, y : Top-left corner of the box in pixels.
        w, h : Box width and height in pixels.
        W, H : Image width and height in pixels.

    Returns:
        (cx, cy, nw, nh) — all values normalised to [0, 1].
    """
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    return cx, cy, w / W, h / H


# ---------------------------------------------------------------------------
# Stage 1 — build
# ---------------------------------------------------------------------------

def stage_build(base_dir: Path = Path("dataset_yolo"), test_size: float = 0.2) -> None:
    """
    Converts the DEArt dataset from COCO JSON to YOLO flat-file format.

    For every sample the function:
      1. Determines the train / val split via stratified sampling.
      2. Saves the PIL image as a JPEG under images/{split}/.
      3. Converts each bounding box from absolute COCO [x,y,w,h] to
         normalised YOLO [cx,cy,w,h] and writes one .txt label file
         under labels/{split}/ (one row per object).
      4. Skips boxes with degenerate dimensions (w or h outside (0, 1]
         after normalisation) to prevent Ultralytics training errors.
      5. Writes dataset.yaml with absolute paths, class count, and
         the full name list so Ultralytics can find everything.

    Output structure:
        {base_dir}/
            dataset.yaml
            images/train/   *.jpg
            images/val/     *.jpg
            labels/train/   *.txt
            labels/val/     *.txt

    Args:
        base_dir  : Root directory for the YOLO dataset.
        test_size : Fraction of data to use as the validation split.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE: build  →  {base_dir}/")
    print(f"{'='*60}\n")

    # ── Create directory structure ──────────────────────────────────
    for split in ("train", "val"):
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Load dataset and create split map ──────────────────────────
    print("Loading dataset…")
    ds = load_dataset("biglam/european_art", split="train")

    indices   = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=42
    )
    split_map = {i: "train" for i in train_idx}
    split_map.update({i: "val" for i in val_idx})

    print(f"  Dataset size : {len(ds):,} samples")
    print(f"  Train        : {len(train_idx):,}")
    print(f"  Val          : {len(val_idx):,}\n")

    # ── Convert and write files ─────────────────────────────────────
    n_images  = 0
    n_labels  = 0
    n_skipped = 0

    for i, sample in enumerate(ds):
        split = split_map[i]
        ann   = json.loads(sample["annotations"])

        img_info = ann["images"][0]
        W, H     = img_info["width"], img_info["height"]

        cat_id2name = {
            cat["id"]: cat["name"].lower().strip()
            for cat in ann.get("categories", [])
        }

        # Save image as JPEG (convert to RGB to drop any alpha channel)
        img_path = base_dir / "images" / split / f"{i:06d}.jpg"
        sample["image"].convert("RGB").save(img_path, quality=95)
        n_images += 1

        # Build YOLO label rows
        rows: list[str] = []
        for obj in ann.get("annotations", []):
            name = cat_id2name.get(obj["category_id"], "")
            cls  = NAME2IDX.get(name)

            if cls is None:
                n_skipped += 1
                continue

            x, y, w, h     = obj["bbox"]
            cx, cy, nw, nh = coco_to_yolo(x, y, w, h, W, H)

            # Guard against malformed annotations (zero-size or out-of-range boxes)
            if not (0 < nw <= 1 and 0 < nh <= 1):
                n_skipped += 1
                continue

            rows.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_labels += 1

        lbl_path = base_dir / "labels" / split / f"{i:06d}.txt"
        lbl_path.write_text("\n".join(rows))

        if i % 500 == 0:
            print(f"  [{i:>6} / {len(ds)}]  images={n_images}  labels={n_labels}  skipped={n_skipped}")

    # ── Write dataset.yaml ──────────────────────────────────────────
    yaml_content = {
        "path":  str(base_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(ALL_CLASSES),
        "names": dict(enumerate(ALL_CLASSES)),
    }
    (base_dir / "dataset.yaml").write_text(
        yaml.dump(yaml_content, allow_unicode=True)
    )

    print(f"\n  Done!")
    print(f"    Images written : {n_images:,}")
    print(f"    Label rows     : {n_labels:,}")
    print(f"    Skipped boxes  : {n_skipped:,}")
    print(f"    YAML written   : {base_dir / 'dataset.yaml'}\n")


# ---------------------------------------------------------------------------
# Stage 2 — augment
# ---------------------------------------------------------------------------

def stage_augment(base_dir: Path = Path("dataset_yolo")) -> None:
    """
    Merges 5,000 images from the COCO val2017 split into the YOLO dataset.

    Why augment with COCO?
    ----------------------
    DEArt labels share many classes with COCO (person, horse, bird …).
    Adding real-world COCO images gives the detector exposure to those
    classes in natural photographic contexts, reducing domain gap for
    the shared classes and improving generalisation.

    What this stage does:
    ---------------------
    1. Downloads COCO val2017 images (~1 GB) into coco_tmp/.
    2. Downloads the matching Ultralytics YOLO-format label files.
    3. Copies images and labels into {base_dir}/images/ and labels/,
       prefixing filenames with "coco_" to avoid collisions.
    4. The existing 80-class COCO label files use indices 0–79, which
       are identical to the first 80 entries of ALL_CLASSES, so no
       remapping is needed.

    Split assignment:
        First 4,000 shuffled images → train
        Remaining 1,000            → val

    Args:
        base_dir: Root directory of the YOLO dataset (must already exist).
    """
    print(f"\n{'='*60}")
    print(f"  STAGE: augment  ({base_dir}/)")
    print(f"{'='*60}\n")

    if not base_dir.exists():
        raise FileNotFoundError(
            f"{base_dir} does not exist. Run --stage build first."
        )

    tmp = Path("coco_tmp")

    # ── Download COCO val2017 ───────────────────────────────────────
    print("Downloading COCO val2017 images (~1 GB)…")
    download(
        "http://images.cocodataset.org/zips/val2017.zip",
        dir=tmp,
    )

    print("Downloading Ultralytics YOLO-format labels…")
    download(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip",
        dir=tmp,
    )

    # ── Shuffle and split ───────────────────────────────────────────
    coco_images = sorted((tmp / "val2017").glob("*.jpg"))
    random.seed(42)
    random.shuffle(coco_images)

    n_train = n_val = 0

    for i, img_path in enumerate(coco_images):  # 5,000 images total
        split = "train" if i < 4000 else "val"
        stem  = img_path.stem

        # Copy image
        shutil.copy(img_path, base_dir / "images" / split / f"coco_{stem}.jpg")

        # Copy matching label (empty file if no annotations for this image)
        lbl_src = tmp / "coco" / "labels" / "val2017" / f"{stem}.txt"
        lbl_dst = base_dir / "labels" / split / f"coco_{stem}.txt"

        if lbl_src.exists():
            shutil.copy(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("")   # Background image — still useful for training

        if split == "train":
            n_train += 1
        else:
            n_val += 1

    print(f"\n  Augmentation complete!")
    print(f"    COCO images added  → train: {n_train}   val: {n_val}")
    print(f"    train total: {len(list((base_dir / 'images' / 'train').iterdir())):,}")
    print(f"    val   total: {len(list((base_dir / 'images' / 'val').iterdir())):,}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="European Art dataset preparation for YOLOv8 training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages (run in this order for a fresh setup):
  inspect   Visualise GT annotations + YOLOv8n predictions on one sample.
  discover  Scan the full dataset and report the class inventory.
  build     Convert DEArt → YOLO flat-file layout + dataset.yaml.
  augment   Merge COCO val2017 images into the YOLO dataset.
  all       Run all four stages in sequence (default).

Examples:
  python data_process.py
  python data_process.py --stage inspect --sample 100
  python data_process.py --stage build --base-dir my_dataset
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["all", "inspect", "discover", "build", "augment"],
        default="all",
        help="Which stage to run (default: all).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=561,
        metavar="IDX",
        help="Dataset index used by the inspect stage (default: 561).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("dataset_yolo"),
        metavar="DIR",
        help="Output directory for the YOLO dataset (default: dataset_yolo/).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Fraction of data reserved for validation (default: 0.20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_all     = args.stage == "all"
    base_dir    = args.base_dir

    if run_all or args.stage == "build":
        stage_build(base_dir=base_dir, test_size=args.val_split)

    if run_all or args.stage == "augment":
        stage_augment(base_dir=base_dir)

    print("\nAll requested stages complete.\n")


if __name__ == "__main__":
    main()
