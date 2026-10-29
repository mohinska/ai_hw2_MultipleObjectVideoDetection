"""Locating the competition data and loading the COCO annotations into dataframes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_INPUT = Path("/kaggle/input")


def find_data_root(hint=None):
    """Return the folder that holds train/, val/, test/ and test_images.csv."""
    candidates = [Path(hint)] if hint else []
    if KAGGLE_INPUT.exists():
        candidates += [p.parent for p in KAGGLE_INPUT.rglob("test_images.csv")]
    candidates += [Path("data"), Path("../data")]
    for root in candidates:
        if (root / "test_images.csv").exists():
            return root.resolve()
    attached = sorted(str(p) for p in KAGGLE_INPUT.glob("*/*")) if KAGGLE_INPUT.exists() else []
    raise FileNotFoundError(
        "Competition data not found. On Kaggle: Add Input -> Competitions -> add the competition. "
        f"Contents of /kaggle/input: {attached or 'empty'}"
    )


def find_input_file(name):
    """Resolve a file attached as Kaggle input by its name, wherever Kaggle mounted it."""
    path = Path(name)
    if path.exists():
        return path
    hits = sorted(KAGGLE_INPUT.rglob(path.name)) if KAGGLE_INPUT.exists() else []
    hits = [h for h in hits if "competitions" not in h.parts]
    if not hits:
        raise FileNotFoundError(f"{path.name} not found under {KAGGLE_INPUT}")
    if len(hits) > 1:
        print("several matches, using the first:", *hits, sep="\n  ")
    return hits[0]


def load_coco(split_dir):
    """Load annotations_coco.json of a split into (images_df, annotations_df).

    annotations_df gets pixel box columns, area, a 0-based `cls` column matching the
    YOLO/submission ids, and the clip name of the frame it belongs to.
    """
    with open(Path(split_dir) / "annotations_coco.json") as f:
        coco = json.load(f)
    images = pd.DataFrame(coco["images"])
    anns = pd.DataFrame(coco["annotations"])
    anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index)
    anns["area"] = anns["w"] * anns["h"]
    anns["cls"] = anns["category_id"] - 1
    anns = anns.merge(
        images[["id", "video", "frame_id", "width", "height"]].rename(columns={"id": "image_id"}),
        on="image_id",
    )
    return images, anns


def subsample_frames(images, step):
    """Keep every `step`-th frame of each clip (frames inside a clip are highly correlated)."""
    images = images.sort_values(["video", "frame_id"])
    rank = images.groupby("video").cumcount()
    return images[rank % step == 0]


def load_split(split_dir, step=1):
    """Frames of a labelled split with their image paths, target boxes and ignored regions.

    Returns (frames_df, targets) where targets[image_id] = (boxes_xyxy, cls, ignore_xyxy),
    all in original pixels; cls is 0-based.
    """
    split_dir = Path(split_dir)
    images, anns = load_coco(split_dir)
    images = subsample_frames(images, step).reset_index(drop=True)
    base = _image_base(split_dir, images["file_name"].iloc[0])
    images["path"] = [str(base / f) for f in images["file_name"]]

    anns = anns[anns["image_id"].isin(images["id"])]
    xyxy = anns[["x", "y", "w", "h"]].to_numpy("float32")
    xyxy[:, 2:] += xyxy[:, :2]
    anns = anns.assign(x1=xyxy[:, 0], y1=xyxy[:, 1], x2=xyxy[:, 2], y2=xyxy[:, 3])
    empty = np.zeros((0, 4), "float32")
    targets = {i: (empty, np.zeros(0, "int64"), empty) for i in images["id"]}
    for image_id, g in anns.groupby("image_id"):
        obj, ign = g[g["iscrowd"] == 0], g[g["iscrowd"] == 1]
        targets[image_id] = (obj[["x1", "y1", "x2", "y2"]].to_numpy("float32"),
                             obj["cls"].to_numpy("int64"),
                             ign[["x1", "y1", "x2", "y2"]].to_numpy("float32"))
    return images, targets


def _image_base(split_dir, file_name):
    """COCO file_name may be relative to the split folder or to the data root."""
    return next(b for b in [split_dir, split_dir.parent, split_dir / "images"] if (b / file_name).exists())


def object_bank(split_dir, classes, min_side=12, step=1):
    """Clean (not occluded, not truncated) objects of `classes` for copy-paste, with their source frame.

    Build it from the training split only, so no validation object is ever pasted into training.
    """
    split_dir = Path(split_dir)
    images, anns = load_coco(split_dir)
    images = subsample_frames(images, step)
    base = _image_base(split_dir, images["file_name"].iloc[0])
    anns = anns[(anns["iscrowd"] == 0) & anns["cls"].isin(classes) & anns["image_id"].isin(images["id"])
                & (anns.get("occlusion", 0) == 0) & (anns.get("truncation", 0) == 0)
                & (np.minimum(anns["w"], anns["h"]) >= min_side)]
    paths = images.set_index("id")["file_name"].map(lambda f: str(base / f))
    return pd.DataFrame({"path": anns["image_id"].map(paths).to_numpy(), "cls": anns["cls"].to_numpy(),
                         "x1": anns["x"].to_numpy(), "y1": anns["y"].to_numpy(),
                         "x2": (anns["x"] + anns["w"]).to_numpy(), "y2": (anns["y"] + anns["h"]).to_numpy(),
                         "long_side": np.maximum(anns["width"], anns["height"]).to_numpy()})


def repeat_factors(frames, targets, threshold):
    """LVIS repeat-factor sampling: r_c = max(1, sqrt(t / f_c)), f_c = share of frames containing class c;
    a frame gets the max r_c of the classes it contains."""
    present = [set(targets[i][1].tolist()) for i in frames["id"]]
    freq = pd.Series([c for s in present for c in s]).value_counts() / len(frames)
    r_cls = np.maximum(1.0, np.sqrt(threshold / freq))
    return np.array([max((r_cls[c] for c in s), default=1.0) for s in present]), r_cls.sort_index()
