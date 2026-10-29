"""Locating the competition data and loading the COCO annotations into dataframes."""

import json
from pathlib import Path

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
