"""Ultralytics models (YOLO, RT-DETR) as train_fn / predict_fn for core.cv.run_cv."""

import os

# silence Ultralytics logs and per-epoch progress bars; must be set before ultralytics is imported
os.environ["YOLO_VERBOSE"] = "False"

import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

from . import CLASSES, SEED  # noqa: E402
from .data import KAGGLE_INPUT, every_nth, image_key  # noqa: E402

settings.update({"wandb": False})


def make_yolo_fns(cfg, work):
    """cfg keys: model, imgsz, epochs, batch, val_step, conf, iou, max_det."""
    work = Path(work)

    def train_fn(train, held_out, tag):
        # the held-out subset only drives Ultralytics' per-epoch curve; the model kept is last.pt,
        # so no epoch is chosen by looking at the held-out fold
        (work / "train.txt").write_text("\n".join(train["path"]))
        (work / "val.txt").write_text("\n".join(every_nth(held_out, cfg["val_step"])["path"]))
        (work / "data.yaml").write_text(yaml.safe_dump({
            "path": str(work), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(CLASSES))}))
        YOLO(cfg["model"]).train(
            data=str(work / "data.yaml"), imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
            seed=SEED, deterministic=True, project=str(work / "runs"), name=tag, exist_ok=True, verbose=False)
        weights = work / f"{tag}.pt"
        shutil.copy(work / "runs" / tag / "weights" / "last.pt", weights)
        return weights

    def predict_fn(weights, paths, chunk=8):
        """Detections in pixel xywh plus mean ms per image (preprocess + inference + NMS)."""
        model = YOLO(weights)
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
        tag = f"{name}_fold{k}"
        hits = [Path(work) / "runs" / tag / "results.csv"]
        hits += sorted(KAGGLE_INPUT.rglob(f"{tag}/results.csv")) if KAGGLE_INPUT.exists() else []
        csv = next((h for h in hits if h.exists()), None)
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
