"""Failure analysis: what happened to every ground-truth object and every confident detection.

A simplified TIDE-style breakdown at IoU 0.5 for detections above a score threshold:
  objects:    detected | wrong_class | poor_box (best IoU 0.1-0.5) | missed
  detections: correct | duplicate | wrong_class | poor_box | in_ignore | background
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from .data import load_coco
from .evaluate import image_key
from .tv import covered_share

SIZE_BINS, SIZE_LABELS = [0, 16, 32, 64, np.inf], ["<16", "16-32", "32-64", ">64"]


def match_detections(split_dir, dets, score_thr=0.3, iou_thr=0.5):
    """Greedy per-frame matching of detections (score >= score_thr) to GT; returns (objects_df, dets_df).

    Sizes are sqrt(area) in original pixels.
    """
    images, anns = load_coco(Path(split_dir))
    key_of = images.set_index("id")["file_name"].map(image_key)
    anns = anns.assign(key=anns["image_id"].map(key_of))
    dets = pd.DataFrame(dets)
    dets = dets[dets["score"] >= score_thr].assign(key=lambda d: d["file_name"].map(
        {f: image_key(f) for f in d["file_name"].unique()}))
    objs_by_key = dict(tuple(anns[anns["iscrowd"] == 0].groupby("key")))
    ign_by_key = dict(tuple(anns[anns["iscrowd"] == 1].groupby("key")))
    obj_rows, det_rows = [], []

    for key, d in dets.groupby("key"):
        d = d.sort_values("score", ascending=False)
        g = objs_by_key.get(key, anns.iloc[:0])
        gb = torch.tensor(np.column_stack([g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]]), dtype=torch.float32)
        db = torch.tensor(np.column_stack([d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]]), dtype=torch.float32)
        iou = box_iou(db, gb).numpy() if len(g) else np.zeros((len(d), 0))
        g_cls, d_cls = g["cls"].to_numpy(), d["cls"].to_numpy()
        taken = np.zeros(len(g), bool)
        ign = ign_by_key.get(key)
        in_ign = covered_share(db.numpy(), np.column_stack([ign["x"], ign["y"], ign["x"] + ign["w"], ign["y"] + ign["h"]])
                               .astype("float32")) >= 0.5 if ign is not None else np.zeros(len(d), bool)

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
        same_cls = np.where(g_cls[None, :] == d_cls[:, None], iou, 0).max(0) if len(d) else np.zeros(len(g))
        obj_rows.append(g.assign(outcome=np.select(
            [taken, best_any >= iou_thr, best_any >= 0.1], ["detected", "wrong_class", "poor_box"], "missed"),
            best_iou=best_any, best_iou_same_class=same_cls))

    seen = set(dets["key"])
    obj_rows.append(anns[(anns["iscrowd"] == 0) & ~anns["key"].isin(seen)].assign(outcome="missed", best_iou=0.0))
    objects = pd.concat(obj_rows, ignore_index=True)
    objects["side"] = np.sqrt(objects["w"] * objects["h"])
    objects["size"] = pd.cut(objects["side"], SIZE_BINS, labels=SIZE_LABELS)
    detections = pd.concat(det_rows, ignore_index=True) if det_rows else dets.assign(outcome=[], best_iou=[])
    return objects, detections


def outcome_tables(objects, detections, classes):
    """Shares of each outcome: objects by size and by class, detections overall."""
    by_size = pd.crosstab(objects["size"], objects["outcome"], normalize="index").round(3)
    by_class = pd.crosstab(objects["cls"].map(dict(enumerate(classes))), objects["outcome"],
                           normalize="index").reindex(classes).round(3)
    det_share = detections["outcome"].value_counts(normalize=True).round(3)
    return by_size, by_class, det_share
