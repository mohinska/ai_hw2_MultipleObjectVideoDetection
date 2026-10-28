"""Writing and checking the Kaggle submission (class score x y w h groups, top-left pixel coords)."""

import pandas as pd

from .evaluate import MAX_DETS


def write_submission(detections, test_csv, out_path):
    """detections: dicts with image_id, cls, score, x, y, w, h. Every test image gets a row."""
    test = pd.read_csv(test_csv)
    det = pd.DataFrame(list(detections), columns=["image_id", "cls", "score", "x", "y", "w", "h"])
    det = det.sort_values("score", ascending=False).groupby("image_id").head(MAX_DETS)

    def fmt(group):
        return " ".join(f"{int(r.cls)} {r.score:.4f} {r.x:.1f} {r.y:.1f} {r.w:.1f} {r.h:.1f}"
                        for r in group.itertuples())

    strings = det.groupby("image_id").apply(fmt) if len(det) else pd.Series(dtype=str)
    sub = pd.DataFrame({"image_id": test["image_id"]})
    sub["PredictionString"] = sub["image_id"].map(strings).fillna("")
    sub.to_csv(out_path, index=False)
    check_submission(out_path, test_csv)
    return sub


def check_submission(path, test_csv):
    """Fail loudly on the mistakes that silently score zero on the leaderboard."""
    sub = pd.read_csv(path, keep_default_na=False)
    test = pd.read_csv(test_csv).set_index("image_id")
    assert list(sub.columns) == ["image_id", "PredictionString"], sub.columns
    assert set(sub["image_id"]) == set(test.index), "image_id set differs from test_images.csv"
    for image_id, s in zip(sub["image_id"], sub["PredictionString"]):
        vals = s.split()
        assert len(vals) % 6 == 0, f"{image_id}: not groups of 6"
        assert len(vals) // 6 <= MAX_DETS, f"{image_id}: more than {MAX_DETS} boxes"
        if vals:
            w, h = test.loc[image_id, ["width", "height"]]
            g = pd.DataFrame([vals[i:i + 6] for i in range(0, len(vals), 6)], dtype=float)
            assert g[0].between(0, 7).all(), f"{image_id}: class id out of range"
            assert (g[2] <= w).all() and (g[3] <= h).all(), f"{image_id}: box outside image (normalised coords?)"
