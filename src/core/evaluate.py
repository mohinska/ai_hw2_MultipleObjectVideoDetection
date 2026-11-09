"""COCO metrics on the pooled ground truth, with VisDrone ignored regions kept as iscrowd=1."""

import contextlib
import io
import json
import tempfile

import numpy as np
import pandas as pd

try:
    # C++ drop-in for pycocotools; the pure-Python evaluator runs out of RAM on ~1M detections
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from . import MAX_DETS

STAT_NAMES = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", f"AR{MAX_DETS}", "AR_small", "AR_medium", "AR_large"]


def coco_evaluate(dets, eval_frames, gt, per_class=False):
    """dets: DataFrame with key, cls (0-based), score, x, y, w, h (pixels).
    eval_frames: frames the model ran on (key, uid); frames without detections count as misses.
    gt: the COCO dict from core.data.load_pool. maxDets is 500, the leaderboard cap.
    """
    ids = set(eval_frames["uid"])
    uid = dict(zip(eval_frames["key"], eval_frames["uid"]))
    dets = pd.DataFrame(dets)
    dets = dets[dets["key"].isin(uid)] if len(dets) else dets
    if dets.empty:
        return {name: 0.0 for name in STAT_NAMES}

    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump({"images": [i for i in gt["images"] if i["id"] in ids],
                   "annotations": [a for a in gt["annotations"] if a["image_id"] in ids],
                   "categories": gt["categories"]}, f)
        f.flush()
        results = np.column_stack([dets["key"].map(uid), dets[["x", "y", "w", "h", "score"]],
                                   dets["cls"] + 1]).astype(np.float64)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(f.name)
            ev = COCOeval(coco_gt, coco_gt.loadRes(results), "bbox")
            ev.params.imgIds = sorted(ids)
            ev.params.maxDets = [1, 10, MAX_DETS]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()

    stats = dict(zip(STAT_NAMES, map(float, ev.stats)))
    if per_class:
        precision = ev.eval["precision"][:, :, :, 0, -1]  # [iou, recall, class, area=all, maxDets=500]
        for k, cat_id in enumerate(ev.params.catIds):
            p = precision[:, :, k]
            stats[f"AP_{coco_gt.cats[cat_id]['name']}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return stats


def per_clip_map(dets, eval_frames, gt):
    """mAP of every clip separately: shows whether a difference between two variants holds across scenes."""
    clip = dets["key"].str.split("/").str[0]
    return {v: coco_evaluate(dets[clip == v], g, gt)["mAP"] for v, g in eval_frames.groupby("video")}
