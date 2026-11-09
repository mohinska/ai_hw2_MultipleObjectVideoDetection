# %% [markdown]
# # RT-DETR+: training on full frames and full-resolution tiles, tiled inference
#
# RT-DETR-L (best single model on the leaderboard, 0.3065 with one full-frame pass) with two changes:
# - **Training data:** every frame is used once whole (shrunk to 1920 px, as before) and as 4 overlapping tiles
#   cut from the original resolution, so small objects are seen at their native size.
# - **Inference:** full frame + flip / larger scale / the same 4 tiles (each tile gets its own 300 queries),
#   merged by WBF or NMS.
#
# Train split only; the official `val` (7 clips) is never trained on, so every inference variant is chosen on it,
# not on the leaderboard. A variant is kept only if it beats the single full pass at full and at 960 px
# resolution and in most clips (the rule of 07_1).
#
# Kaggle: GPU T4 x2, Internet on, competition data attached. Runtime ~7 h (training capped by `HOURS`).

# %%
HOURS = 5.5       # training budget; Ultralytics fits the LR schedule into it
DEVICE = [0, 1]   # both T4s (DDP); use 0 if DDP fails at the start of training
TRAIN_STEP = 8    # every 8th train frame -> ~3,000 frames x 5 images (full + 4 tiles)
BATCH = 8         # total over the GPUs; 4 if CUDA runs out of memory
NAME = "rtdetr_tiles"

# %% [markdown]
# ## Setup

# %%
import subprocess
import sys
from pathlib import Path

if Path("../src/core").exists():
    sys.path.insert(0, "../src")
else:
    repo = Path("/tmp/repo")  # outside /kaggle/working so it is not saved as output
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/mohinska/ai_hw2_MultipleObjectVideoDetection", str(repo)], check=True)
    sys.path.insert(0, str(repo / "src"))

# %%
# !pip install -q ultralytics faster-coco-eval

# %%
import json

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from core import CLASSES, SEED
from core.cv import environment
from core.data import covered_share, every_nth, find_data_root, image_key, load_pool, pool_targets
from core.evaluate import coco_evaluate, per_clip_map
from core.submission import write_submission
from core.temporal import rescore
from core.variants import TILE_SHARE, choose_variant, collect_view, compare_variants, variant_dets
from core.yolo import load_ultralytics

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../outputs/rtdetr_tiles")
WORK.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/rtdetr_tiles")
print(torch.cuda.device_count(), "GPU(s)")
environment()

# %% [markdown]
# ## Data
# Train: train split. Hold-out: every 5th frame of the official val clips.

# %%
DATA = find_data_root()
frames, gt = load_pool(DATA)
targets = pool_targets(gt)
train = every_nth(frames[frames["split"] == "train"], TRAIN_STEP)
val = every_nth(frames[frames["split"] == "val"], 5)
print(len(train), "train frames,", len(val), "val frames,", val["video"].nunique(), "val clips")

# %% [markdown]
# ## Training images: full frame + 4 tiles
# Ignored regions are filled with the frame's mean colour and objects >= 50% inside them are dropped (as in
# `core.yolo.write_masked`). In a tile, a box is kept if >= 50% of it lies inside the tile, clipped to its edge.
# The tiles are the ones `core.variants` uses at inference (60% of the frame side, 2 x 2).

# %%
def crops_of(w, h, tiles):
    out = [(0, 0, w, h)]
    if tiles:
        tw, th = round(w * TILE_SHARE), round(h * TILE_SHARE)
        out += [(x, y, tw, th) for y in (0, h - th) for x in (0, w - tw)]
    return out


def write_images(frames, out_dir, tiles, max_side=1920):
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    paths = []
    for r in frames.itertuples():
        img = cv2.imread(r.path)
        h, w = img.shape[:2]
        boxes, cls, ignore = targets[r.uid]
        if len(ignore):
            fill = img.reshape(-1, 3).mean(0).astype(np.uint8)
            for x1, y1, x2, y2 in ignore.astype(int):
                img[max(y1, 0):y2, max(x1, 0):x2] = fill
            visible = 1 - covered_share(boxes, ignore)
            boxes, cls = boxes[visible >= 0.5], cls[visible >= 0.5]
        area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
        for i, (x0, y0, cw, ch) in enumerate(crops_of(w, h, tiles)):
            dst = out_dir / "images" / f"{Path(r.path).parent.name}_{Path(r.path).stem}_{i}.jpg"
            paths.append(str(dst))
            if dst.exists():
                continue
            b = boxes - [x0, y0, x0, y0]
            c = np.clip(b, 0, [cw, ch, cw, ch])
            keep = ((c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) / area >= 0.5) & (c[:, 2] - c[:, 0] >= 2) & (c[:, 3] - c[:, 1] >= 2)
            c, k = c[keep] / [cw, ch, cw, ch], cls[keep]
            crop = img[y0:y0 + ch, x0:x0 + cw]
            if max(cw, ch) > max_side:
                s = max_side / max(cw, ch)
                crop = cv2.resize(crop, (round(cw * s), round(ch * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_dir / "labels" / f"{dst.stem}.txt").write_text("\n".join(
                f"{j} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}" for j, (x1, y1, x2, y2) in zip(k, c)))
    return paths


train_paths = write_images(train, TMP / "train", tiles=True)
val_paths = write_images(every_nth(val, 4), TMP / "val", tiles=False)  # only drives the per-epoch curve
(TMP / "train.txt").write_text("\n".join(train_paths))
(TMP / "val.txt").write_text("\n".join(val_paths))
(TMP / "data.yaml").write_text(yaml.safe_dump(
    {"path": str(TMP), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(CLASSES))}))
print(len(train_paths), "training images,", len(val_paths), "curve images")

# %% [markdown]
# ## Training
# Settings of the 0.3065 run (AdamW, lr 3e-4, cosine, 1280 px) with a time budget instead of epochs.
# Ultralytics writes last.pt to /kaggle/working after every epoch, so a session cut off still leaves weights.
# The model kept is last.pt.

# %%
weights = WORK / f"{NAME}.pt"
found = weights if weights.exists() else next(iter(Path("/kaggle/input").rglob(f"{NAME}.pt")), None) \
    if Path("/kaggle/input").exists() else None
if found is None:
    model = load_ultralytics("rtdetr-l.pt")
    model.train(data=str(TMP / "data.yaml"), imgsz=1280, epochs=100, time=HOURS, batch=BATCH, device=DEVICE,
                optimizer="AdamW", lr0=3e-4, weight_decay=1e-4, cos_lr=True, close_mosaic=2, workers=4,
                seed=SEED, deterministic=True, project=str(WORK / "runs"), name=NAME, exist_ok=True, verbose=False)
    import shutil
    shutil.copy(WORK / "runs" / NAME / "weights" / "last.pt", weights)
    found = weights
print("weights:", found)

# %%
log = pd.read_csv(WORK / "runs" / NAME / "results.csv").rename(columns=str.strip) \
    if (WORK / "runs" / NAME / "results.csv").exists() else None
log[["epoch", "time", "metrics/mAP50-95(B)"]].round(4) if log is not None else "no training log (weights reused)"

# %% [markdown]
# ## Inference variants on val
# Views: full frame at 1280, its flip, full frame at 1600, 2 x 2 tiles at 1280. All again on frames shrunk to
# 960 px, like the low-resolution test clips.

# %%
model = load_ultralytics(found)


def detector(imgsz):
    def detect(images):  # RGB arrays -> (xyxy, cls, score) per image
        res = model.predict([cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images], imgsz=imgsz, conf=0.001,
                            max_det=300, half=True, verbose=False)
        return [(b.xyxy.reshape(-1, 4), b.cls.astype(int), b.conf) for b in (r.boxes.cpu().numpy() for r in res)]
    return detect


VIEW_SPECS = {"full": (1280, "full"), "flip": (1280, "flip"), "full1600": (1600, "full"), "tiles": (1280, "tiles")}
REF = "full"
VARIANTS = {
    REF: dict(views=["full"]),
    "flip_wbf": dict(views=["full", "flip"], iou=0.65, fuse="wbf"),
    "scale_wbf": dict(views=["full", "full1600"], iou=0.65, fuse="wbf"),
    "tiles_nms": dict(views=["full", "tiles"], iou=0.65),
    "tiles_wbf": dict(views=["full", "tiles"], iou=0.65, fuse="wbf"),
    "tiles_flip_wbf": dict(views=["full", "flip", "tiles"], iou=0.65, fuse="wbf"),
    "all_nms": dict(views=["full", "flip", "full1600", "tiles"], iou=0.65),
    "all_wbf": dict(views=["full", "flip", "full1600", "tiles"], iou=0.65, fuse="wbf"),
}


def collect(paths, names, max_side=None):
    return {v: collect_view(detector(VIEW_SPECS[v][0]), paths, VIEW_SPECS[v][1], max_side=max_side) for v in names}


native_views = collect(val["path"].tolist(), VIEW_SPECS)
low_views = collect(val["path"].tolist(), VIEW_SPECS, max_side=960)
for v in VIEW_SPECS:
    native_views[v].to_parquet(WORK / f"val_view_{NAME}_{v}.parquet")
    low_views[v].to_parquet(WORK / f"val_view_{NAME}_{v}_lowres.parquet")

native = compare_variants(native_views, VARIANTS, val, gt, REF)
lowres = compare_variants(low_views, VARIANTS, val, gt, REF, clips=False)
CHOSEN = choose_variant(native, lowres, REF)
print("chosen:", CHOSEN, f"(clips: {native.attrs['clips']})")
table = native.join(lowres[["mAP"]].rename(columns={"mAP": "mAP_960"}))
table.to_csv(WORK / f"val_variants_{NAME}.csv")
table

# %% [markdown]
# ## Temporal rescoring on val
# Neighbours are 5 frames apart (every 5th val frame); on test, where every frame is predicted, `step=5` keeps
# the same distance in time. Kept only if it beats no rescoring in most clips.

# %%
val_dets = variant_dets(native_views, VARIANTS[CHOSEN])
val_dets.to_parquet(WORK / f"val_dets_{NAME}.parquet")
keys = val["key"].tolist()
base_clips = pd.Series(per_clip_map(val_dets, val, gt))
rows = {"none": {"mAP": coco_evaluate(val_dets, val, gt)["mAP"], "clips_better": 0}}
OPTIONS_T = {f"window {w}, alpha {a}": dict(window=w, alpha=a) for w in (1, 2) for a in (0.3, 0.5)}
for label, o in OPTIONS_T.items():
    d = rescore(val_dets, keys, o["window"], 0.5, o["alpha"])
    rows[label] = {"mAP": coco_evaluate(d, val, gt)["mAP"],
                   "clips_better": int((pd.Series(per_clip_map(d, val, gt)) > base_clips).sum())}
temporal = pd.DataFrame(rows).T
ok = temporal.drop("none")
ok = ok[(ok["mAP"] > temporal.loc["none", "mAP"]) & (ok["clips_better"] > len(base_clips) / 2)]
TEMPORAL = OPTIONS_T[ok["mAP"].idxmax()] if len(ok) else None
print("temporal:", TEMPORAL)
(WORK / f"chosen_{NAME}.json").write_text(json.dumps({"variant": CHOSEN, "temporal": TEMPORAL}, indent=1))
temporal.round(4)

# %% [markdown]
# ## Test submission
# Only the views of the chosen variant. Detections before and after rescoring are saved for the final fusion.

# %%
test = pd.read_csv(DATA / "test_images.csv")
test_paths = [str(DATA / f) for f in test["file_name"]]
test_views = collect(test_paths, VARIANTS[CHOSEN]["views"])
test_dets = variant_dets(test_views, VARIANTS[CHOSEN])
test_dets.to_parquet(WORK / f"test_dets_{NAME}.parquet")
if TEMPORAL:
    test_dets = rescore(test_dets, [image_key(p) for p in test_paths], TEMPORAL["window"], 0.5, TEMPORAL["alpha"],
                        step=5)
    test_dets.to_parquet(WORK / f"test_dets_{NAME}_temporal.parquet")
write_submission(test_dets, test, WORK / f"submission_{NAME}.csv").head()
