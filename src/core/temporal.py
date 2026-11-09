"""Temporal rescoring inside a clip: frames are consecutive video, so a real object shows up in the
neighbouring frames and a false positive usually does not."""

import re

import numpy as np
import pandas as pd

from .fusion import iou_matrix


def clip_neighbours(keys, window, step=1):
    """{key: up to `window` keys before and after it in the same clip, `step` places apart}; never crosses clips.
    Keys are '<clip>/<frame>' (core.data.image_key). step=5 on every frame of a clip gives the same neighbours
    in time as step=1 on every 5th frame."""
    frames = pd.DataFrame({"k": list(keys)})
    frames["clip"] = frames["k"].str.split("/").str[0]
    frames["n"] = frames["k"].map(lambda k: int(re.sub(r"\D", "", k.split("/")[-1]) or 0))
    out = {}
    for _, g in frames.sort_values(["clip", "n"]).groupby("clip"):
        names = g["k"].tolist()
        for i, k in enumerate(names):
            out[k] = [names[i + d * step] for d in range(-window, window + 1)
                      if d and 0 <= i + d * step < len(names)]
    return out


def rescore(dets, keys, window=1, iou=0.5, alpha=0.5, step=1):
    """score <- (1 - alpha) * score + alpha * mean over the neighbouring frames of the best score of a same-class
    box with IoU >= iou there (0 if there is none or the frame has no detections).

    dets: key, cls, score, x, y, w, h. keys: every frame the model ran on, also those without detections.
    A stable object keeps its score, a box seen in one frame only loses `alpha` of it.
    Boxes are linked by IoU only, without a motion model: objects that move by more than their size per frame
    lose the link.
    """
    dets = dets.reset_index(drop=True)
    xy = dets[["x", "y", "w", "h"]].to_numpy("float32")
    boxes = np.column_stack([xy[:, :2], xy[:, :2] + xy[:, 2:]])
    cls = dets["cls"].to_numpy()
    score = dets["score"].to_numpy("float32")
    rows = dets.groupby("key").indices
    new = score.copy()
    for k, nbrs in clip_neighbours(keys, window, step).items():
        if k not in rows or not nbrs:
            continue
        i = rows[k]
        seen = np.zeros(len(i), "float32")
        for nb in nbrs:
            if nb in rows:
                j = rows[nb]
                hit = (iou_matrix(boxes[i], boxes[j]) >= iou) & (cls[i][:, None] == cls[j][None])
                seen += (hit * score[j][None]).max(1)
        new[i] = (1 - alpha) * score[i] + alpha * seen / len(nbrs)
    return dets.assign(score=new)
