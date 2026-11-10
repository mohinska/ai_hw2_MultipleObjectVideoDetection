"""Grouped K-fold cross-validation with out-of-fold (OOF) predictions, shared by every model family.

A model plugs in two functions:
    train_fn(train_frames, held_out_frames, tag) -> weights path
    predict_fn(weights, image_paths) -> (detections DataFrame [key, cls, score, x, y, w, h], ms per image)
"""

import json
import platform
import shutil
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
    """A file from this run (work dir) or from an earlier run attached as Kaggle input, else None.
    A file found in the input is copied to `work`, so a resumed run's output is complete on its own."""
    local = Path(work) / name
    if local.exists():
        return local
    hit = next(iter(sorted(KAGGLE_INPUT.rglob(name))), None) if KAGGLE_INPUT.exists() else None
    if hit is None:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hit, local)
    return local


def run_cv(frames, gt, cfg, train_fn, predict_fn, work, n_folds=N_FOLDS, folds=None, retrain=()):
    """Train on all folds but one, predict every frame of the held-out fold, for each fold.

    `folds` limits the run to some held-out folds (e.g. [0] for a hyperparameter sweep); the "oof" metrics
    then cover only those folds' frames. Finished folds ({name}_fold{k}.pt + OOF parquet + metrics json,
    here or in /kaggle/input) are reused, so a run cut off by the Kaggle time limit can be resumed;
    folds listed in `retrain` ignore saved files and are trained again (e.g. a fold whose training failed).
    Writes cv_{name}.json with cfg, environment and all metrics.
    """
    import torch
    if KAGGLE_INPUT.exists() and not torch.cuda.is_available():
        raise RuntimeError("No GPU: set Accelerator to GPU T4 in the notebook settings")
    work = Path(work)
    frames = frames.assign(fold=frames["video"].map(assign_folds(frames, n_folds)))
    folds = list(range(n_folds)) if folds is None else list(folds)
    fold_metrics, oof_parts = {}, []
    for k in folds:
        train = every_nth(frames[frames["fold"] != k], cfg["train_step"])
        held_out = frames[frames["fold"] == k]
        tag = f"{cfg['name']}_fold{k}"
        print(f"fold {k}: {len(train)} train frames, {len(held_out)} held-out frames "
              f"({held_out['video'].nunique()} clips)")

        if k in retrain:
            weights = train_fn(train, held_out, tag)
        else:
            weights = existing(f"{tag}.pt", work) or train_fn(train, held_out, tag)
        oof_file, metrics_file = existing(f"oof_{tag}.parquet", work), existing(f"metrics_{tag}.json", work)
        if k not in retrain and oof_file and metrics_file:
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
    oof_stats = coco_evaluate(oof, frames[frames["fold"].isin(folds)], gt, per_class=True)
    (work / f"cv_{cfg['name']}.json").write_text(json.dumps(
        {"cfg": cfg, "n_folds": n_folds, "folds_run": folds, "seed": SEED, "env": environment(),
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


def compare(names, work, fold=None, cols=("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                                          "AR500", "ms_per_image")):
    """One row per run: metrics of held-out `fold` (metrics_{name}_fold{k}.json), or the OOF metrics of
    cv_{name}.json when fold is None. Runs whose files are not found are skipped with a note."""
    rows = {}
    for name in names:
        f = existing(f"metrics_{name}_fold{fold}.json" if fold is not None else f"cv_{name}.json", work)
        if f is None:
            print(f"{name}: not run yet")
            continue
        m = json.loads(f.read_text())
        if fold is None:
            if len(m.get("folds_run", m["folds"])) < m["n_folds"]:
                print(f"{name}: only fold(s) {m['folds_run']} run, no full OOF")
                continue
            m = {**m["oof"], "ms_per_image": pd.DataFrame(m["folds"]).T["ms_per_image"].mean()}
        rows[name] = {c: m.get(c) for c in cols}
    return pd.DataFrame(rows).T.astype(float)


def load_oof(name, folds, work):
    """Saved OOF detections of a run for the given held-out folds."""
    return pd.concat([pd.read_parquet(existing(f"oof_{name}_fold{k}.parquet", work)) for k in folds],
                     ignore_index=True)


def best_per_family(prefixes, work, fold=0):
    """{prefix: name of the run with the best mAP on held-out `fold`} among metrics_{name}_fold{fold}.json
    found in `work` or /kaggle/input; families without a run are left out."""
    files = list(Path(work).glob(f"metrics_*_fold{fold}.json"))
    files += list(KAGGLE_INPUT.rglob(f"metrics_*_fold{fold}.json")) if KAGGLE_INPUT.exists() else []
    scores = {f.name[len("metrics_"):-len(f"_fold{fold}.json")]: json.loads(f.read_text())["mAP"] for f in files}
    best = {}
    for prefix in prefixes:
        family = {n: m for n, m in scores.items() if n.startswith(prefix)}
        if family:
            best[prefix] = max(family, key=family.get)
    return best
