"""Weighted Boxes Fusion (Solovyev et al., arXiv 1910.13302): overlapping boxes are averaged, not discarded."""

import numpy as np
import pandas as pd


def iou_matrix(a, b):
    """IoU of every xyxy box in `a` with every one in `b`."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def wbf(det, iou, n_models):
    """Fuse det (file_name, cls, score, x, y, w, h) per image and class.

    Boxes go in by descending score and join the cluster whose fused box they overlap by more than `iou`.
    A fused box is the score-weighted mean of its members and its score their mean, scaled by
    min(members, n_models) / n_models: a box that only one of the n_models sources found is down-weighted.
    """
    parts = []
    for (name, cls), g in det.groupby(["file_name", "cls"], sort=False):
        xy = g[["x", "y", "w", "h"]].to_numpy("float64")
        box = np.column_stack([xy[:, :2], xy[:, :2] + xy[:, 2:]])
        score = g["score"].to_numpy("float64")
        n = len(g)
        fused, bsum = np.empty((n, 4)), np.zeros((n, 4))
        wsum, cnt = np.zeros(n), np.zeros(n)
        k = 0
        for i in np.argsort(-score):
            j = k
            if k:
                ious = iou_matrix(box[i][None], fused[:k])[0]
                if ious.max() > iou:
                    j = ious.argmax()
            if j == k:
                k += 1
            bsum[j] += score[i] * box[i]
            wsum[j] += score[i]
            cnt[j] += 1
            fused[j] = bsum[j] / wsum[j]
        parts.append(pd.DataFrame({
            "x": fused[:k, 0], "y": fused[:k, 1], "w": fused[:k, 2] - fused[:k, 0], "h": fused[:k, 3] - fused[:k, 1],
            "cls": cls, "score": wsum[:k] / cnt[:k] * np.minimum(cnt[:k], n_models) / n_models, "file_name": name}))
    out = pd.concat(parts, ignore_index=True).astype({"x": "float32", "y": "float32", "w": "float32", "h": "float32",
                                                      "cls": "int8", "score": "float32"})
    return out[["x", "y", "w", "h", "cls", "score", "file_name"]]
