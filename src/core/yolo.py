"""Ultralytics models (YOLO, RT-DETR) as train_fn / predict_fn for core.cv.run_cv."""

import os

# silence Ultralytics logs and per-epoch progress bars; must be set before ultralytics is imported
os.environ["YOLO_VERBOSE"] = "False"

import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from ultralytics import RTDETR, YOLO, settings  # noqa: E402

from . import CLASSES, SEED  # noqa: E402
from .data import KAGGLE_INPUT, covered_share, every_nth, image_key  # noqa: E402

settings.update({"wandb": False})

P2_CFG = Path(__file__).parent / "yolo11s-p2.yaml"  # YOLO11s with an extra P2/4 level for tiny objects


def write_masked(frames, targets, out_dir, max_side=1920):
    """Copies of `frames` with ignored regions filled grey and YOLO labels without objects >= 50% inside them.

    Ultralytics cannot mask regions itself, so it trains on these copies (written once, reused across folds).
    Returns the copies' paths in the order of `frames`.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    paths = []
    for r in frames.itertuples():
        src = Path(r.path)
        dst = out_dir / "images" / f"{src.parent.name}_{src.stem}.jpg"
        paths.append(str(dst))
        if dst.exists():
            continue
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        boxes, cls, ignore = targets[r.uid]
        if len(ignore):
            fill = img.reshape(-1, 3).mean(0).astype(np.uint8)
            for x1, y1, x2, y2 in ignore.astype(int):
                img[max(y1, 0):y2, max(x1, 0):x2] = fill
            visible = 1 - covered_share(boxes, ignore)
            boxes, cls = boxes[visible >= 0.5], cls[visible >= 0.5]
        if max(h, w) > max_side:  # Ultralytics resizes to 1280 anyway; smaller copies load faster
            s = max_side / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b = np.clip(boxes / [w, h, w, h], 0, 1)
        (out_dir / "labels" / f"{dst.stem}.txt").write_text("\n".join(
            f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}" for c, (x1, y1, x2, y2) in zip(cls, b)))
    return paths


def load_ultralytics(path):
    """YOLO or RT-DETR from weights or a model yaml; RT-DETR files have "rtdetr" in their name."""
    return RTDETR(str(path)) if "rtdetr" in Path(str(path)).name else YOLO(str(path))


def _results_csv(tag, work):
    hits = [Path(work) / "runs" / tag / "results.csv"]
    hits += sorted(KAGGLE_INPUT.rglob(f"{tag}/results.csv")) if KAGGLE_INPUT.exists() else []
    return next((h for h in hits if h.exists()), None)


def make_yolo_fns(cfg, work, targets=None):
    """cfg keys: model (pretrained weights), imgsz, epochs, batch, val_step, conf, iou, max_det; optional:
    model_cfg (architecture yaml, e.g. P2_CFG; pretrained layers that fit are loaded from `model`),
    mask_ignore (train on masked copies, needs `targets` from core.data.pool_targets),
    train_args (extra Ultralytics training arguments, e.g. {"cls": 1.0} or {"time": 1.5} for a budget in hours),
    denoising (RT-DETR: number of denoising queries during training, 0 switches denoising off).
    The model kept per fold is last.pt."""
    work = Path(work)

    def train_fn(train, held_out, tag):
        # the held-out subset only drives Ultralytics' per-epoch curve; the model kept is last.pt,
        # so no epoch is chosen by looking at the held-out fold
        train_paths = (write_masked(train, targets, Path("/tmp/masked_frames")) if cfg.get("mask_ignore")
                       else train["path"].tolist())
        (work / "train.txt").write_text("\n".join(train_paths))
        (work / "val.txt").write_text("\n".join(every_nth(held_out, cfg["val_step"])["path"]))
        (work / "data.yaml").write_text(yaml.safe_dump({
            "path": str(work), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(CLASSES))}))
        model = (load_ultralytics(cfg["model_cfg"]).load(cfg["model"]) if cfg.get("model_cfg")
                 else load_ultralytics(cfg["model"]))
        if "denoising" in cfg:
            def set_denoising(trainer):
                trainer.model.model[-1].num_denoising = cfg["denoising"]
            model.add_callback("on_train_start", set_denoising)
        model.train(
            data=str(work / "data.yaml"), imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
            seed=SEED, deterministic=True, project=str(work / "runs"), name=tag, exist_ok=True, verbose=False,
            **cfg.get("train_args", {}))
        weights = work / f"{tag}.pt"
        shutil.copy(work / "runs" / tag / "weights" / "last.pt", weights)
        return weights

    def predict_fn(weights, paths, chunk=8):
        """Detections in pixel xywh plus mean ms per image (preprocess + inference + NMS)."""
        model = load_ultralytics(weights)
        parts, ms = [], []
        paths = [str(p) for p in paths]
        # warm-up pass (CUDA init, cudnn autotune) so the first images do not inflate the timing
        model.predict(paths[:1], imgsz=cfg["imgsz"], verbose=False)
        for i in range(0, len(paths), chunk):  # a list source is decoded up front, so feed small chunks
            batch = paths[i:i + chunk]
            results = model.predict(batch, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"],
                                    max_det=cfg["max_det"], stream=True, verbose=False)
            for path, res in zip(batch, results):
                ms.append(sum(res.speed.values()))
                b = res.boxes.cpu().numpy()
                xyxy = b.xyxy.reshape(-1, 4)
                parts.append(pd.DataFrame({
                    "key": image_key(path), "cls": b.cls.astype("int8"), "score": b.conf.astype("float32"),
                    "x": xyxy[:, 0], "y": xyxy[:, 1], "w": xyxy[:, 2] - xyxy[:, 0], "h": xyxy[:, 3] - xyxy[:, 1]}))
        return pd.concat(parts, ignore_index=True), float(np.mean(ms))

    return train_fn, predict_fn


def plot_curves(name, n_folds, work):
    """Held-out mAP and training loss per epoch for every fold, from Ultralytics' results.csv."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for k in range(n_folds):
        csv = _results_csv(f"{name}_fold{k}", work)
        if csv is None:
            print(f"fold {k}: no results.csv (weights reused without their run folder)")
            continue
        log = pd.read_csv(csv).rename(columns=str.strip)
        axes[0].plot(log["epoch"], log["metrics/mAP50-95(B)"], label=f"fold {k}")
        axes[1].plot(log["epoch"], log.filter(like="train/").sum(axis=1), label=f"fold {k}")
    axes[0].set(xlabel="epoch", ylabel="mAP50-95", title="Held-out mAP during training (every 10th frame)")
    axes[1].set(xlabel="epoch", ylabel="sum of train losses", title="Training loss")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.savefig(Path(work) / f"curves_{name}.png", dpi=110, bbox_inches="tight")
    return fig


def plot_budget_curves(names, fold, work):
    """Held-out mAP against training hours for several runs on one fold (Ultralytics' results.csv),
    so detectors with different epoch lengths are compared at the same compute."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in names:
        csv = _results_csv(f"{name}_fold{fold}", work)
        if csv is None:
            print(f"{name}: no results.csv for fold {fold}")
            continue
        log = pd.read_csv(csv).rename(columns=str.strip)
        ax.plot(log["time"] / 3600, log["metrics/mAP50-95(B)"], marker="o", ms=3, label=name)
    ax.set(xlabel="training time, hours (T4)", ylabel="mAP50-95 (every 10th held-out frame)",
           title=f"Held-out mAP vs compute, fold {fold}")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(Path(work) / f"budget_curves_fold{fold}.png", dpi=110)
    return fig
