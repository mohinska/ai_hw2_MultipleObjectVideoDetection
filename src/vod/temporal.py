"""Temporal rescoring of detections inside a clip: frames are consecutive video, so a real object shows up
in the neighbouring frames and a false positive usually does not."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .fusion import iou_matrix


def clip_neighbours(file_names, window):
    """{frame: frames up to `window` places before or after it in the same clip}; never crosses clips."""
    frames = pd.DataFrame({"f": [str(f) for f in file_names]})
    frames["clip"] = frames["f"].map(lambda f: Path(f).parent.name)  # '<clip>/<frame>.jpg', as evaluate.image_key
    frames["n"] = frames["f"].map(lambda f: int(re.sub(r"\D", "", Path(f).stem) or 0))
    out = {}
    for _, g in frames.sort_values(["clip", "n"]).groupby("clip"):
        names = g["f"].tolist()
        for i, f in enumerate(names):
            out[f] = names[max(0, i - window):i] + names[i + 1:i + 1 + window]
    return out


def rescore(dets, file_names, window=1, iou=0.5, alpha=0.5):
    """score <- (1 - alpha) * score + alpha * mean over the neighbouring frames of the best score of a same-class
    box with IoU >= iou there (0 if there is none or the frame has no detections).

    dets: file_name, cls, score, x, y, w, h. file_names: every frame of the split, also those without detections.
    A stable object keeps or gains score, a box seen in one frame only loses (1 - alpha) of it.
    """
    # link by IoU between neighbouring frames, no motion model; objects that move by more than
    # their size per frame lose the link. Add a tracker (Kalman / IoU tracks) only if val shows that.
    dets = dets.reset_index(drop=True)
    xy = dets[["x", "y", "w", "h"]].to_numpy("float32")
    boxes = np.column_stack([xy[:, :2], xy[:, :2] + xy[:, 2:]])
    cls = dets["cls"].to_numpy()
    score = dets["score"].to_numpy("float32")
    rows = dets.groupby("file_name").indices
    new = score.copy()
    for f, nbrs in clip_neighbours(file_names, window).items():
        if f not in rows or not nbrs:
            continue
        i = rows[f]
        seen = np.zeros(len(i), "float32")
        for nb in nbrs:
            if nb in rows:
                j = rows[nb]
                hit = (iou_matrix(boxes[i], boxes[j]) >= iou) & (cls[i][:, None] == cls[j][None])
                seen += (hit * score[j][None]).max(1)
        new[i] = (1 - alpha) * score[i] + alpha * seen / len(nbrs)
    return dets.assign(score=new)
