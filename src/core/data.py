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


def load_coco(split_dir):
    """annotations_coco.json of one split as (images, annotations) DataFrames; annotations get pixel x, y, w, h,
    area, a 0-based `cls` and the clip, frame number and size of their frame."""
    coco = json.loads((Path(split_dir) / "annotations_coco.json").read_text())
    images, anns = pd.DataFrame(coco["images"]), pd.DataFrame(coco["annotations"])
    anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index)
    anns["area"], anns["cls"] = anns["w"] * anns["h"], anns["category_id"] - 1
    anns = anns.merge(images[["id", "video", "frame_id", "width", "height"]].rename(columns={"id": "image_id"}),
                      on="image_id")
    return images, anns


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


def pool_targets(gt):
    """uid -> (object boxes xyxy, cls 0-based, ignored regions xyxy), all float32/int64 in original pixels."""
    anns = pd.DataFrame(gt["annotations"])
    xyxy = np.array(anns["bbox"].tolist(), dtype="float32")
    xyxy[:, 2:] += xyxy[:, :2]
    anns = anns.assign(x1=xyxy[:, 0], y1=xyxy[:, 1], x2=xyxy[:, 2], y2=xyxy[:, 3])
    empty = np.zeros((0, 4), "float32")
    targets = {img["id"]: (empty, np.zeros(0, "int64"), empty) for img in gt["images"]}
    cols = ["x1", "y1", "x2", "y2"]
    for uid, g in anns.groupby("image_id"):
        obj, ign = g[g["iscrowd"] == 0], g[g["iscrowd"] == 1]
        targets[uid] = (obj[cols].to_numpy("float32"), (obj["category_id"] - 1).to_numpy("int64"),
                        ign[cols].to_numpy("float32"))
    return targets


def covered_share(boxes, regions):
    """Share of each xyxy box covered by xyxy `regions` (overlaps counted twice, clipped to 1)."""
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes), "float32")
    ix = np.clip(np.minimum(boxes[:, None, 2], regions[None, :, 2])
                 - np.maximum(boxes[:, None, 0], regions[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, None, 3], regions[None, :, 3])
                 - np.maximum(boxes[:, None, 1], regions[None, :, 1]), 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.clip((ix * iy).sum(1) / np.maximum(area, 1e-6), 0, 1)
