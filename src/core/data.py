"""Labelled frames of train + val as one pool, and the grouped folds every notebook uses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import N_FOLDS

KAGGLE_INPUT = Path("/kaggle/input")


def find_data_root():
    """Folder that holds train/, val/, test/ and test_images.csv."""
    candidates = [p.parent for p in KAGGLE_INPUT.rglob("test_images.csv")] if KAGGLE_INPUT.exists() else []
    for root in candidates + [Path("data"), Path("../data")]:
        if (root / "test_images.csv").exists():
            return root.resolve()
    raise FileNotFoundError("Competition data not found: Add Input -> Competitions")


def image_key(path):
    """'.../<clip>/<frame>.jpg' -> '<clip>/<frame>': the join key between frames, GT and predictions."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def load_pool(data_root):
    """All labelled frames (train + val) and their COCO ground truth with globally unique image ids.

    frames: one row per frame with path, key (clip/frame), uid (id in `gt`), split, video, frame_id.
    gt: COCO dict; ignored regions stay iscrowd=1 annotations.
    """
    frames, gt_images, gt_anns, offset = [], [], [], 0
    for split in ["train", "val"]:
        split_dir = Path(data_root) / split
        coco = json.loads((split_dir / "annotations_coco.json").read_text())
        images = pd.DataFrame(coco["images"])
        base = next(b for b in [split_dir, split_dir.parent, split_dir / "images"]
                    if (b / images["file_name"].iloc[0]).exists())
        images["path"] = [str(base / f) for f in images["file_name"]]
        images["key"] = images["path"].map(image_key)
        images["uid"] = images["id"] + offset
        images["split"] = split
        frames.append(images)
        gt_images += [{**img, "id": img["id"] + offset} for img in coco["images"]]
        gt_anns += [{**a, "image_id": a["image_id"] + offset} for a in coco["annotations"]]
        offset = int(images["uid"].max()) + 1
    for i, a in enumerate(gt_anns):
        a["id"] = i + 1
    gt = {"images": gt_images, "annotations": gt_anns, "categories": coco["categories"]}
    return pd.concat(frames, ignore_index=True), gt


def assign_folds(frames, n_folds=N_FOLDS):
    """clip -> fold. Clips cut from one flight (uav0000013_00000_v, uav0000013_01073_v) share a fold."""
    frames = frames.sort_values(["video", "frame_id"])
    source = frames["video"].str.split("_").str[0]
    fold = np.empty(len(frames), dtype=int)
    for k, (_, idx) in enumerate(GroupKFold(n_splits=n_folds).split(frames, groups=source)):
        fold[idx] = k
    return pd.Series(fold, index=frames["video"].to_numpy()).groupby(level=0).first()


def every_nth(frames, step):
    """Every `step`-th frame of each clip (neighbouring frames are near-duplicates)."""
    frames = frames.sort_values(["video", "frame_id"])
    return frames[frames.groupby("video").cumcount() % step == 0]
