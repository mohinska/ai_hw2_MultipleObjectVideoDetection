"""Writing and checking the Kaggle submission."""

import numpy as np
import pandas as pd

from . import MAX_DETS
from .data import image_key


def write_submission(dets, test, out_path):
    """dets: key, cls, score, x, y, w, h (pixels). test: test_images.csv as a DataFrame.

    One row per test image: 'cls score x y w h ...', top-left corner in original pixels, <= 500 boxes.
    """
    dets = pd.DataFrame(dets)
    dets["image_id"] = dets["key"].map(dict(zip(test["file_name"].map(image_key), test["image_id"])))
    dets = dets.sort_values("score", ascending=False).groupby("image_id").head(MAX_DETS)
    strings = dets.groupby("image_id")[["cls", "score", "x", "y", "w", "h"]].apply(
        lambda g: " ".join(f"{int(r.cls)} {r.score:.4f} {r.x:.1f} {r.y:.1f} {r.w:.1f} {r.h:.1f}" for r in g.itertuples()))
    sub = pd.DataFrame({"image_id": test["image_id"]})
    sub["PredictionString"] = sub["image_id"].map(strings).fillna("")
    sub.to_csv(out_path, index=False)
    check_submission(out_path, test)
    return sub


def check_submission(path, test):
    """Fail loudly on the mistakes that silently score zero on the leaderboard."""
    sub = pd.read_csv(path, keep_default_na=False)
    assert list(sub.columns) == ["image_id", "PredictionString"], sub.columns
    assert set(sub["image_id"]) == set(test["image_id"]), "image_id set differs from test_images.csv"
    size = test.set_index("image_id")[["width", "height"]]
    for image_id, s in zip(sub["image_id"], sub["PredictionString"]):
        g = np.array(s.split(), dtype=float).reshape(-1, 6)
        assert len(g) <= MAX_DETS, f"{image_id}: more than {MAX_DETS} boxes"
        assert ((g[:, 0] >= 0) & (g[:, 0] <= 7)).all(), f"{image_id}: class id out of range"
        assert (g[:, 2] <= size.loc[image_id, "width"]).all() and (g[:, 3] <= size.loc[image_id, "height"]).all(), \
            f"{image_id}: box outside image (normalised coords?)"
