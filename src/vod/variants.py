"""Inference variants of a trained detector (flip, tiles, merging), compared on val before a submission."""

import time

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms

from .evaluate import MAX_DETS, coco_evaluate, per_clip_map
from .tv import _device, resize_long_side

TILE_SHARE = 0.6  # side of a tile relative to the frame: 2 x 2 tiles overlapping by 20% of the frame
EDGE_MARGIN = 2  # px; a box this close to an inner tile edge is probably a cut object


def torchvision_detector(model, cfg):
    """detect(images): list of RGB arrays -> list of (xyxy, cls 0-based, score) in the pixels of each array."""
    device = _device()
    model.to(device).eval()

    @torch.no_grad()
    def detect(images):
        resized = [resize_long_side(im, cfg["imgsz"]) for im in images]
        tensors = [torch.from_numpy(im).permute(2, 0, 1).float().div(255).to(device) for im, _ in resized]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outs = model(tensors)
        results = []
        for out, (_, scale) in zip(outs, resized):
            keep = out["labels"] > 0
            results.append((out["boxes"][keep].float().cpu().numpy() / scale,
                            out["labels"][keep].cpu().numpy() - 1,
                            out["scores"][keep].float().cpu().numpy()))
        return results
    return detect


def yolo_detector(model, imgsz=1280, conf=0.001, iou=0.7, max_det=MAX_DETS):
    """The same detect(images) for an Ultralytics model."""
    def detect(images):
        bgr = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images]
        results = model.predict(bgr, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy())
                for r in results]
    return detect


def _crops(img, view):
    """(array, dx, dy) of each crop that `view` runs the detector on."""
    h, w = img.shape[:2]
    if view == "full":
        return [(img, 0, 0)]
    if view == "flip":
        return [(img[:, ::-1].copy(), 0, 0)]
    if view == "tiles":
        tw, th = round(w * TILE_SHARE), round(h * TILE_SHARE)
        return [(img[y:y + th, x:x + tw], x, y) for y in (0, h - th) for x in (0, w - tw)]
    raise ValueError(view)


def _to_frame(boxes, view, dx, dy, crop_shape, frame_shape):
    """Mask of boxes to keep and the boxes in frame pixels."""
    ch, cw = crop_shape[:2]
    fh, fw = frame_shape[:2]
    b = boxes.reshape(-1, 4).astype("float32").copy()
    keep = np.ones(len(b), bool)
    if view == "flip":
        b[:, [0, 2]] = fw - b[:, [2, 0]]
    if view == "tiles":
        if dx > 0:
            keep &= b[:, 0] >= EDGE_MARGIN
        if dx + cw < fw:
            keep &= b[:, 2] <= cw - EDGE_MARGIN
        if dy > 0:
            keep &= b[:, 1] >= EDGE_MARGIN
        if dy + ch < fh:
            keep &= b[:, 3] <= ch - EDGE_MARGIN
        b[:, [0, 2]] += dx
        b[:, [1, 3]] += dy
    return keep, b


def collect_view(detect, paths, view, max_side=None, batch=4):
    """Detections of one view in original-image pixels (file_name, cls, score, x, y, w, h).

    max_side shrinks the frames first, to imitate the low-resolution test clips on val.
    """
    parts, pending = [], []
    t0 = time.perf_counter()

    def flush():
        for (path, _, dx, dy, shape, frame_shape, scale), (boxes, cls, score) in zip(pending, detect([p[1] for p in pending])):
            keep, b = _to_frame(boxes, view, dx, dy, shape, frame_shape)
            b = b[keep] / scale
            part = pd.DataFrame({"x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]},
                                dtype="float32")
            part["cls"] = np.asarray(cls)[keep].astype("int8")
            part["score"] = np.asarray(score)[keep].astype("float32")
            part["file_name"] = path
            parts.append(part)
        pending.clear()

    for path in paths:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        scale = 1.0
        if max_side and max(img.shape[:2]) > max_side:
            img, scale = resize_long_side(img, max_side)
        for crop, dx, dy in _crops(img, view):
            pending.append((path, crop, dx, dy, crop.shape, img.shape, scale))
            if len(pending) >= batch:
                flush()
    if pending:
        flush()
    dets = pd.concat(parts, ignore_index=True)
    dets.attrs["ms_per_frame"] = round(1000 * (time.perf_counter() - t0) / len(paths), 1)
    return dets


def merge_views(views, iou=None, max_det=MAX_DETS):
    """Class-wise NMS over all views (iou=None keeps the boxes as they are) and the top max_det per image."""
    det = pd.concat(views, ignore_index=True)
    if iou is not None and len(det):
        xy = det[["x", "y", "w", "h"]].to_numpy()
        boxes = torch.from_numpy(np.column_stack([xy[:, :2], xy[:, :2] + xy[:, 2:]]))
        group = pd.factorize(det["file_name"])[0] * 100 + det["cls"].to_numpy().astype(int)
        keep = batched_nms(boxes, torch.from_numpy(det["score"].to_numpy()), torch.from_numpy(group), iou)
        det = det.iloc[keep.numpy()]
    return det.sort_values("score", ascending=False).groupby("file_name").head(max_det).reset_index(drop=True)


def compare_variants(views, variants, gt_json, paths, ref, clips=True):
    """mAP of every variant ({name: dict(views=[...], iou=...)}), and in how many clips it beats `ref`."""
    rows, per_clip = {}, {}
    for name, v in variants.items():
        dets = merge_views([views[n] for n in v["views"]], v.get("iou"))
        m = coco_evaluate(gt_json, dets, file_names=paths)
        rows[name] = {k: round(m[k], 4) for k in ["mAP", "AP50", "AP75", "AP_small", "AR500"]}
        rows[name]["ms_per_frame"] = sum(views[n].attrs["ms_per_frame"] for n in v["views"])
        if clips:
            per_clip[name] = per_clip_map(gt_json, dets, paths)
    table = pd.DataFrame(rows).T
    if clips:
        per_clip = pd.DataFrame(per_clip).T
        table["clips_better"] = (per_clip > per_clip.loc[ref]).sum(axis=1)
        table.attrs["clips"] = per_clip.shape[1]
    return table


def choose_variant(native, lowres, ref):
    """Best variant that beats `ref` at both resolutions and in most clips; `ref` if there is none."""
    ok = (native["mAP"] > native.loc[ref, "mAP"]) & (lowres["mAP"] > lowres.loc[ref, "mAP"]) \
        & (native["clips_better"] > native.attrs["clips"] / 2)
    if not ok.any():
        return ref
    return ((native["mAP"] + lowres["mAP"]) / 2)[ok].idxmax()
