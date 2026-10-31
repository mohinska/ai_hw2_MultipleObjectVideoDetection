"""Ultralytics YOLO under the common protocol: masked training copy, training and evaluation."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from . import CLASSES, SEED
from .evaluate import coco_evaluate, per_clip_map
from .tv import augment, covered_share, resize_long_side
from .variants import collect_view, yolo_detector

P2_CFG = Path(__file__).parent / "yolo11s-p2.yaml"


def write_dataset(frames, targets, out_dir, degrade_p=0.0, max_side=1920, seed=SEED):
    """Copy of the frames with ignored regions masked and YOLO labels; returns the list of images.

    Objects at least half inside an ignored region are dropped, as in vod.tv.FrameDataset.
    degrade_p imitates the low-resolution test clips, once per frame since Ultralytics has no such augmentation.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    images = []
    for r in frames.itertuples():
        src = Path(r.path)
        dst = out_dir / "images" / f"{src.parent.name}_{src.stem}.jpg"
        images.append(str(dst))
        if dst.exists():
            continue
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        boxes, cls, ignore = targets[r.id]
        if len(ignore):
            fill = img.reshape(-1, 3).mean(0).astype(np.uint8)
            for x1, y1, x2, y2 in ignore.astype(int):
                img[max(y1, 0):y2, max(x1, 0):x2] = fill
            visible = 1 - covered_share(boxes, ignore)
            boxes, cls = boxes[visible >= 0.5], cls[visible >= 0.5]
        if np.random.random() < degrade_p:
            img, _ = augment(img, np.zeros((0, 4), "float32"), {"hflip": 0.0, "degrade_p": 1.0})
        if max(h, w) > max_side:
            img, _ = resize_long_side(img, max_side)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b = boxes / [w, h, w, h]
        lines = [f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                 for c, (x1, y1, x2, y2) in zip(cls, np.clip(b, 0, 1))]
        (out_dir / "labels" / f"{dst.stem}.txt").write_text("\n".join(lines))
    return images


def write_data_yaml(path, train_images, val_images):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_txt, val_txt = path.with_suffix(".train.txt"), path.with_suffix(".val.txt")
    train_txt.write_text("\n".join(train_images))
    val_txt.write_text("\n".join(val_images))
    path.write_text(yaml.safe_dump({"path": str(path.parent), "train": str(train_txt), "val": str(val_txt),
                                    "names": dict(enumerate(CLASSES))}))
    return path


def build_yolo(cfg):
    """Pretrained weights, or a model config with the pretrained layers that fit (P2 head)."""
    if cfg.get("model"):
        model = YOLO(str(cfg["model"]))
        return model.load(cfg["weights"]) if cfg.get("weights") else model
    return YOLO(cfg["weights"])


def train_yolo(name, cfg, data_yaml, work):
    """Train under `cfg`; keys other than model, weights and degrade_p go to Ultralytics as they are."""
    args = {k: v for k, v in cfg.items() if k not in ("model", "weights", "degrade_p")}
    model = build_yolo(cfg)
    model.train(data=str(data_yaml), seed=SEED, deterministic=True, project=str(Path(work) / "runs"), name=name,
                exist_ok=True, **args)
    return Path(work) / "runs" / name


def evaluate_yolo(name, cfg, run_dir, val_frames, val_gt, weights="last.pt"):
    """The same evaluation as for the torchvision detectors, saved as metrics.json and history.csv."""
    run_dir = Path(run_dir)
    detect = yolo_detector(YOLO(run_dir / "weights" / weights), imgsz=cfg["imgsz"])
    paths = val_frames["path"].tolist()
    dets = collect_view(detect, paths, "full")
    metrics = coco_evaluate(val_gt, dets, file_names=paths, per_class=True)
    metrics.update(name=name, weights=weights, ms_per_image=round(batch1_speed(detect, paths), 1),
                   per_clip_mAP=per_clip_map(val_gt, dets, paths), config={k: str(v) for k, v in cfg.items()})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    results = pd.read_csv(run_dir / "results.csv").rename(columns=str.strip)
    results.rename(columns={"metrics/mAP50-95(B)": "mAP", "metrics/mAP50(B)": "AP50"}).to_csv(
        run_dir / "history.csv", index=False)
    dets.to_parquet(run_dir / "val_dets.parquet")
    return metrics


def batch1_speed(detect, paths, n=100, warmup=10):
    """Median time per image at batch 1, image loading excluded."""
    times = []
    for i, path in enumerate(paths[:n + warmup]):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        detect([img])
        if i >= warmup or len(paths) <= warmup:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))
