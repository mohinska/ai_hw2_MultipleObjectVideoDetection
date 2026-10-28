"""COCO-style evaluation that respects VisDrone ignored regions.

Ignored regions are stored in annotations_coco.json as iscrowd=1 annotations, so
pycocotools already skips detections that fall inside them. We raise maxDets to 500
because that is the per-image cap used by the leaderboard and scenes are dense.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    # C++ drop-in for pycocotools; the pure-Python one runs out of RAM on ~1M detections
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = 500
STAT_NAMES = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", f"AR{MAX_DETS}", "AR_small", "AR_medium", "AR_large"]


def image_key(file_name):
    """'.../images/<clip>/<frame>.jpg' -> '<clip>/<frame>', the join key between GT and predictions."""
    p = Path(file_name)
    return f"{p.parent.name}/{p.stem}"


def coco_evaluate(gt_json, detections, file_names=None, per_class=False):
    """Evaluate detections against a split's annotations_coco.json.

    detections: DataFrame (or list of dicts) with file_name, cls (0-based), score, x, y, w, h (pixels).
    file_names: the frames the model was run on, so frames without detections still count
    as misses; defaults to every frame of the clips that appear in `detections`.
    Returns a dict of the 12 standard COCO stats (and per-class AP if requested).
    """
    det = pd.DataFrame(detections)
    if det.empty:
        return {name: 0.0 for name in STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
    key_to_id = {image_key(img["file_name"]): img["id"] for img in gt.dataset["images"]}

    file_to_id = {f: key_to_id[image_key(f)] for f in det["file_name"].unique()}
    # loadRes accepts an N x 7 array [image_id, x, y, w, h, score, category_id]
    results = np.column_stack([det["file_name"].map(file_to_id), det[["x", "y", "w", "h", "score"]],
                               det["cls"] + 1]).astype(np.float64)

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        if file_names is not None:
            ev.params.imgIds = sorted(key_to_id[image_key(f)] for f in file_names)
        else:
            clips = {k.split("/")[0] for k in map(image_key, file_to_id)}
            ev.params.imgIds = sorted(i for k, i in key_to_id.items() if k.split("/")[0] in clips)
        ev.params.maxDets = [1, 10, MAX_DETS]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = dict(zip(STAT_NAMES, ev.stats.tolist()))

    if per_class:
        # precision: [iou, recall, class, area, maxDets]; area 0 = all, maxDets -1 = MAX_DETS
        precision = ev.eval["precision"][:, :, :, 0, -1]
        for k, cat_id in enumerate(ev.params.catIds):
            p = precision[:, :, k]
            stats[f"AP_{gt.cats[cat_id]['name']}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return stats


def boxes_xyxy_to_xywh(xyxy):
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    return np.column_stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
