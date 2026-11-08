"""Weighted Boxes Fusion (Solovyev et al., arXiv 1910.13302) and the ensemble of the fold models.

Detections of several models cannot be averaged directly (each model returns its own list of boxes),
so boxes of the same class that overlap are first clustered, then each cluster becomes one box.
"""

import numpy as np
import pandas as pd

from . import N_FOLDS
from .cv import existing


def iou_matrix(a, b):
    """IoU of every xyxy box in `a` with every one in `b`."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def wbf(dets, n_models, iou=0.55):
    """Fuse detections (key, cls, score, x, y, w, h) of `n_models` models, per image and class.

    Boxes go in by descending score and join the cluster whose fused box they overlap by more than `iou`.
    A fused box is the score-weighted mean of its members; its score is the mean member score times
    min(members, n_models) / n_models, so a box found by only one of four models keeps a quarter of its score.
    """
    parts = []
    for (key, cls), g in dets.groupby(["key", "cls"], sort=False):
        xywh = g[["x", "y", "w", "h"]].to_numpy("float64")
        box = np.column_stack([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]])
        score = g["score"].to_numpy("float64")
        n = len(g)
        fused, box_sum = np.empty((n, 4)), np.zeros((n, 4))
        score_sum, count = np.zeros(n), np.zeros(n)
        k = 0  # number of clusters so far
        for i in np.argsort(-score):
            j = k  # new cluster unless the box overlaps an existing one
            if k:
                ious = iou_matrix(box[i][None], fused[:k])[0]
                if ious.max() > iou:
                    j = ious.argmax()
            if j == k:
                k += 1
            box_sum[j] += score[i] * box[i]
            score_sum[j] += score[i]
            count[j] += 1
            fused[j] = box_sum[j] / score_sum[j]
        parts.append(pd.DataFrame({
            "key": key, "cls": cls,
            "score": score_sum[:k] / count[:k] * np.minimum(count[:k], n_models) / n_models,
            "x": fused[:k, 0], "y": fused[:k, 1], "w": fused[:k, 2] - fused[:k, 0], "h": fused[:k, 3] - fused[:k, 1]}))
    return pd.concat(parts, ignore_index=True)


def fold_ensemble(name, predict_fn, paths, work, n_folds=N_FOLDS, iou=0.55):
    """Every fold model ({name}_fold{k}.pt) predicts `paths`; the predictions are fused by WBF.

    Together the fold models have seen every labelled clip, while each single one missed a quarter of them.
    The models are evaluated out-of-fold; the gain of the ensemble is measured on the leaderboard.
    """
    dets = []
    for k in range(n_folds):
        d, _ = predict_fn(existing(f"{name}_fold{k}.pt", work), paths)
        dets.append(d)
        print(f"fold {k} model: {len(d)} boxes")
    return wbf(pd.concat(dets, ignore_index=True), n_folds, iou)
