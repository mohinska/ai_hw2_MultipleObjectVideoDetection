"""Weighted Boxes Fusion of finished Kaggle submission files (no GPU, no model weights).

Usage (from the repo root):
    uv run --with pandas --with numpy --with scikit-learn python experiments/fuse_submissions.py \
        --subs a.csv b.csv c.csv --weights 2 1 1 --iou 0.6 --test test_images.csv --out fused.csv [--rank]

--rank replaces each submission's scores by their percentile within that file before fusion, so that models with
different score scales (RT-DETR vs YOLO) count equally. Each output row keeps the top 500 boxes per image.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import MAX_DETS  # noqa: E402
from core.fusion import wbf  # noqa: E402


def read_submission(path):
    """image_id, PredictionString -> one row per box: key (image_id), cls, score, x, y, w, h."""
    sub = pd.read_csv(path, keep_default_na=False)
    parts = []
    for image_id, s in zip(sub["image_id"], sub["PredictionString"]):
        g = np.array(s.split(), dtype=float).reshape(-1, 6)
        parts.append(pd.DataFrame({"key": image_id, "cls": g[:, 0].astype(int), "score": g[:, 1],
                                   "x": g[:, 2], "y": g[:, 3], "w": g[:, 4], "h": g[:, 5]}))
    return pd.concat(parts, ignore_index=True), sub["image_id"].tolist()


def write(dets, image_ids, out):
    dets = dets.sort_values("score", ascending=False).groupby("key").head(MAX_DETS)
    strings = dets.groupby("key")[["cls", "score", "x", "y", "w", "h"]].apply(
        lambda g: " ".join(f"{int(r.cls)} {r.score:.4f} {r.x:.1f} {r.y:.1f} {r.w:.1f} {r.h:.1f}" for r in g.itertuples()))
    sub = pd.DataFrame({"image_id": image_ids})
    sub["PredictionString"] = sub["image_id"].map(strings).fillna("")
    sub.to_csv(out, index=False)
    return sub


def check(path, test_csv):
    """Same checks as core.submission.check_submission, keyed by image_id."""
    sub = pd.read_csv(path, keep_default_na=False)
    test = pd.read_csv(test_csv).set_index("image_id")
    assert set(sub["image_id"]) == set(test.index), "image_id set differs from test_images.csv"
    for image_id, s in zip(sub["image_id"], sub["PredictionString"]):
        g = np.array(s.split(), dtype=float).reshape(-1, 6)
        assert len(g) <= MAX_DETS and ((g[:, 0] >= 0) & (g[:, 0] <= 7)).all(), image_id
        assert (g[:, 2] <= test.loc[image_id, "width"]).all() and (g[:, 3] <= test.loc[image_id, "height"]).all(), image_id
    print("submission OK:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    parts, ids = [], None
    for i, path in enumerate(a.subs):
        dets, image_ids = read_submission(path)
        ids = ids or image_ids
        assert set(image_ids) == set(ids), f"{path}: different image ids"
        if a.rank:
            dets["score"] = dets["score"].rank(pct=True)
        print(f"{path}: {len(dets)} boxes, {len(dets) / len(image_ids):.0f} per image, "
              f"median score {dets['score'].median():.3f}")
        parts.append(dets.assign(src=i))
    fused = wbf(pd.concat(parts, ignore_index=True), len(a.subs), a.iou, a.weights)
    write(fused, ids, a.out)
    print(f"{a.out}: {len(fused)} fused boxes before the 500-per-image cap")
    if a.test:
        check(a.out, a.test)


if __name__ == "__main__":
    main()
