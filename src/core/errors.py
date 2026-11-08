"""Failure analysis: what happened to every ground-truth object and every confident detection.

A simplified TIDE-style breakdown at IoU 0.5 for detections above a score threshold:
  objects:    detected | wrong_class | poor_box (best IoU 0.1-0.5) | missed
  detections: correct | duplicate | wrong_class | poor_box | in_ignore | background
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from . import CLASSES
from .data import covered_share

SIZE_BINS, SIZE_LABELS = [0, 16, 32, 64, np.inf], ["<16", "16-32", "32-64", ">64"]


def _iou(a, b):
    """IoU matrix of xyxy boxes a (n, 4) and b (m, 4)."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-6)


def _xyxy(df):
    return np.column_stack([df["x"], df["y"], df["x"] + df["w"], df["y"] + df["h"]]).astype("float32")


def match_detections(dets, frames, gt, score_thr=0.3, iou_thr=0.5):
    """Greedy per-frame matching of detections (score >= score_thr) to the GT of `frames`.

    Returns (objects, detections) with an `outcome` column; sizes are sqrt(area) in original pixels.
    """
    key_of = dict(zip(frames["uid"], frames["key"]))
    anns = pd.DataFrame([a for a in gt["annotations"] if a["image_id"] in key_of])
    anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index)
    anns = anns.assign(key=anns["image_id"].map(key_of), cls=anns["category_id"] - 1)
    dets = pd.DataFrame(dets)
    dets = dets[(dets["score"] >= score_thr) & dets["key"].isin(set(frames["key"]))]
    objs_by_key = dict(tuple(anns[anns["iscrowd"] == 0].groupby("key")))
    ign_by_key = dict(tuple(anns[anns["iscrowd"] == 1].groupby("key")))
    obj_rows, det_rows = [], []

    for key, d in dets.groupby("key"):
        d = d.sort_values("score", ascending=False)
        g = objs_by_key.get(key, anns.iloc[:0])
        iou = _iou(_xyxy(d), _xyxy(g)) if len(g) else np.zeros((len(d), 0))
        g_cls, d_cls = g["cls"].to_numpy(), d["cls"].to_numpy()
        taken = np.zeros(len(g), bool)
        ign = ign_by_key.get(key)
        in_ign = covered_share(_xyxy(d), _xyxy(ign)) >= 0.5 if ign is not None else np.zeros(len(d), bool)

        outcome = []
        for i in range(len(d)):
            same = np.where(g_cls == d_cls[i], iou[i], 0) if len(g) else np.zeros(0)
            free = np.where(taken, 0, same) if len(g) else same
            if len(g) and free.max() >= iou_thr:
                taken[int(free.argmax())] = True
                outcome.append("correct")
            elif len(g) and same.max() >= iou_thr:
                outcome.append("duplicate")
            elif len(g) and iou[i].max() >= iou_thr:
                outcome.append("wrong_class")
            elif len(g) and iou[i].max() >= 0.1:
                outcome.append("poor_box")
            else:
                outcome.append("in_ignore" if in_ign[i] else "background")
        det_rows.append(d.assign(outcome=outcome, best_iou=iou.max(1) if len(g) else 0.0))

        best_any = iou.max(0) if len(d) else np.zeros(len(g))
        obj_rows.append(g.assign(outcome=np.select(
            [taken, best_any >= iou_thr, best_any >= 0.1], ["detected", "wrong_class", "poor_box"], "missed"),
            best_iou=best_any))

    seen = set(dets["key"])
    obj_rows.append(anns[(anns["iscrowd"] == 0) & ~anns["key"].isin(seen)].assign(outcome="missed", best_iou=0.0))
    objects = pd.concat(obj_rows, ignore_index=True)
    objects["side"] = np.sqrt(objects["w"] * objects["h"])
    objects["size"] = pd.cut(objects["side"], SIZE_BINS, labels=SIZE_LABELS)
    detections = pd.concat(det_rows, ignore_index=True) if det_rows else dets.assign(outcome=[], best_iou=[])
    return objects, detections


def outcome_tables(objects, detections):
    """Shares of each outcome: objects by size and by class, detections overall."""
    by_size = pd.crosstab(objects["size"], objects["outcome"], normalize="index").round(3)
    by_class = pd.crosstab(objects["cls"].map(dict(enumerate(CLASSES))), objects["outcome"],
                           normalize="index").reindex(CLASSES).round(3)
    det_share = detections["outcome"].value_counts(normalize=True).round(3)
    return by_size, by_class, det_share


def worst_frames(objects, detections):
    """Frame with the most missed objects, the most wrong-class objects and the most background false positives."""
    per_frame = objects.groupby("key")["outcome"].value_counts().unstack(fill_value=0)
    fp = detections[detections["outcome"] == "background"].groupby("key").size()
    cases = {"most missed objects": per_frame["missed"].idxmax() if "missed" in per_frame else None,
             "most wrong-class objects": per_frame["wrong_class"].idxmax() if "wrong_class" in per_frame else None,
             "most background false positives": fp.idxmax() if len(fp) else None}
    return {title: key for title, key in cases.items() if key is not None}


def plot_failures(cases, objects, detections, frames, out_path=None):
    """Missed objects red, wrong class orange, poor box yellow, detected green; background false positives blue."""
    path_of = dict(zip(frames["key"], frames["path"]))
    colors = {"missed": "red", "wrong_class": "orange", "poor_box": "yellow", "detected": "lime"}
    fig, axes = plt.subplots(len(cases), 1, figsize=(16, 9 * len(cases)), squeeze=False)
    for ax, (title, key) in zip(axes[:, 0], cases.items()):
        ax.imshow(Image.open(path_of[key]))
        for r in objects[objects["key"] == key].itertuples():
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, ec=colors[r.outcome], lw=0.8))
        for r in detections[(detections["key"] == key) & (detections["outcome"] == "background")].itertuples():
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, ec="deepskyblue", lw=0.8))
        ax.set_title(f"{title}: {key}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=100)
    return fig
