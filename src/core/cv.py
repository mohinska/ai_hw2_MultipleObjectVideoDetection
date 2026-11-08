"""Grouped K-fold cross-validation with out-of-fold (OOF) predictions, shared by every model family.

A model plugs in two functions:
    train_fn(train_frames, held_out_frames, tag) -> weights path
    predict_fn(weights, image_paths) -> (detections DataFrame [key, cls, score, x, y, w, h], ms per image)
"""

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from . import N_FOLDS, SEED
from .data import KAGGLE_INPUT, assign_folds, every_nth
from .evaluate import coco_evaluate


def environment(packages=("torch", "torchvision", "ultralytics", "faster-coco-eval", "pycocotools",
                           "scikit-learn", "pandas", "numpy")):
    """Python, installed package versions and GPU: what a reported number was produced with."""
    env = {"python": platform.python_version()}
    for p in packages:
        try:
            env[p] = version(p)
        except PackageNotFoundError:
            pass
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        env["gpu"] = "unknown"
    return env


def existing(name, work):
    """A file from this run (work dir) or from an earlier run attached as Kaggle input, else None."""
    hits = [Path(work) / name] + (sorted(KAGGLE_INPUT.rglob(name)) if KAGGLE_INPUT.exists() else [])
    return next((h for h in hits if h.exists()), None)


def run_cv(frames, gt, cfg, train_fn, predict_fn, work, n_folds=N_FOLDS):
    """Train on all folds but one, predict every frame of the held-out fold, for each fold.

    Finished folds ({name}_fold{k}.pt + OOF parquet + metrics json, here or in /kaggle/input) are reused,
    so a run cut off by the Kaggle time limit can be resumed. Writes cv_{name}.json with cfg and all metrics.
    """
    work = Path(work)
    frames = frames.assign(fold=frames["video"].map(assign_folds(frames, n_folds)))
    fold_metrics, oof_parts = {}, []
    for k in range(n_folds):
        train = every_nth(frames[frames["fold"] != k], cfg["train_step"])
        held_out = frames[frames["fold"] == k]
        tag = f"{cfg['name']}_fold{k}"
        print(f"fold {k}: {len(train)} train frames, {len(held_out)} held-out frames "
              f"({held_out['video'].nunique()} clips)")

        weights = existing(f"{tag}.pt", work) or train_fn(train, held_out, tag)
        oof_file, metrics_file = existing(f"oof_{tag}.parquet", work), existing(f"metrics_{tag}.json", work)
        if oof_file and metrics_file:
            oof, m = pd.read_parquet(oof_file), json.loads(metrics_file.read_text())
        else:
            oof, ms = predict_fn(weights, held_out["path"].tolist())
            oof.to_parquet(work / f"oof_{tag}.parquet")
            m = {**coco_evaluate(oof, held_out, gt, per_class=True), "ms_per_image": ms}
            (work / f"metrics_{tag}.json").write_text(json.dumps(m, indent=1))
        oof_parts.append(oof)
        fold_metrics[k] = m
        print(f"  mAP {m['mAP']:.4f}  AP50 {m['AP50']:.4f}  AP_small {m['AP_small']:.4f}")

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_stats = coco_evaluate(oof, frames, gt, per_class=True)
    (work / f"cv_{cfg['name']}.json").write_text(json.dumps(
        {"cfg": cfg, "n_folds": n_folds, "seed": SEED, "env": environment(),
         "folds": fold_metrics, "oof": oof_stats}, indent=1))
    return {"folds": fold_metrics, "oof": oof_stats, "oof_dets": oof}


def cv_table(result, cols=("mAP", "AP50", "AP_small", "AP_medium", "AP_large", "ms_per_image")):
    """Per-fold rows, mean and std across folds, and the OOF row (all frames evaluated at once)."""
    cols = list(cols)
    table = pd.DataFrame(result["folds"]).T[cols].rename(index=lambda k: f"fold {k}")
    n = len(table)
    table.loc["mean"] = table.mean()
    table.loc["std"] = table.iloc[:n].std()
    table.loc["OOF"] = pd.Series({c: result["oof"].get(c) for c in cols})
    table.loc["OOF", "ms_per_image"] = table.loc["mean", "ms_per_image"]
    return table
