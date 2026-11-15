# %% [markdown]
# # RT-DETR+ on all labelled flights, fused with the earlier submissions
#
# Same recipe as `r_rtdetr_tiles` (RT-DETR-L, full frames + full-resolution 2 x 2 tiles, AdamW 3e-4, cosine),
# with two changes:
# - **Training data:** train **and** val clips. Cross-flight generalisation is the bottleneck (the curves of every
#   model plateau after a few epochs), so the 7 val clips (new flights) are worth more than another hold-out.
#   There is no hold-out left, so the inference variant is not chosen here: it is `all_nms` (full + flip +
#   1600 px + tiles, NMS 0.65), the variant `r_rtdetr_tiles` chose on val (beat the full pass in 7/7 clips).
# - **Seed** differs from the first run, so the two models make different mistakes and can be fused.
#
# Fusion (end of the notebook, if the outputs are attached as input):
# - R1 = `test_dets_rtdetr_tiles.parquet` (output of `r_rtdetr_tiles`, public 0.3293),
# - Y = `submission_final.csv` (output of `07_2_ensemble`, public 0.3215).
#
# Kaggle: GPU T4 x2, Internet on, competition data + the two outputs above attached. Runtime ~9.5 h.

# %%
HOURS = 7.0       # training budget; Ultralytics fits the LR schedule into it
DEVICE = [0, 1]   # both T4s (DDP); use 0 if DDP fails at the start of training
TRAIN_STEP = 8    # every 8th frame -> ~4,000 frames x 5 images (full + 4 tiles)
BATCH = 8         # total over the GPUs; 4 if CUDA runs out of memory
NAME = "rtdetr_all"
FUSE_IOU = 0.65   # same NMS IoU as the view fusion chosen on val

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
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from core import CLASSES, SEED
from core.cv import environment
from core.data import covered_share, every_nth, find_data_root, image_key, load_pool, pool_targets
from core.submission import write_submission
from core.variants import TILE_SHARE, collect_view, merge_views, variant_dets
from core.yolo import load_ultralytics

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../outputs/rtdetr_all")
WORK.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/rtdetr_all")
INPUT = Path("/kaggle/input")
print(torch.cuda.device_count(), "GPU(s)")
environment()

# %% [markdown]
# ## Data
# Every 8th frame of every labelled clip (train + val). Ultralytics needs a validation set for its per-epoch curve:
# a few val frames are used, which are also trained on, so that curve is not a hold-out score.

# %%
DATA = find_data_root()
frames, gt = load_pool(DATA)
targets = pool_targets(gt)
train = every_nth(frames, TRAIN_STEP)
curve = every_nth(frames[frames["split"] == "val"], 40)
print(len(train), "train frames from", train["video"].nunique(), "clips;", len(curve), "curve frames")

# %% [markdown]
# ## Training images: full frame + 4 tiles
# As in `r_rtdetr_tiles`: ignored regions are filled with the frame's mean colour and objects >= 50% inside them
# are dropped; in a tile, a box is kept if >= 50% of it lies inside the tile, clipped to its edge.

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
curve_paths = write_images(curve, TMP / "val", tiles=False)
(TMP / "train.txt").write_text("\n".join(train_paths))
(TMP / "val.txt").write_text("\n".join(curve_paths))
(TMP / "data.yaml").write_text(yaml.safe_dump(
    {"path": str(TMP), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(CLASSES))}))
print(len(train_paths), "training images,", len(curve_paths), "curve images")

# %% [markdown]
# ## Training
# Settings of `r_rtdetr_tiles`, another seed. The model kept is last.pt (written after every epoch, so a session
# cut off still leaves weights; a finished run attached as input is reused).

# %%
weights = WORK / f"{NAME}.pt"
found = weights if weights.exists() else next(iter(INPUT.rglob(f"{NAME}.pt")), None) if INPUT.exists() else None
if found is None:
    model = load_ultralytics("rtdetr-l.pt")
    model.train(data=str(TMP / "data.yaml"), imgsz=1280, epochs=100, time=HOURS, batch=BATCH, device=DEVICE,
                optimizer="AdamW", lr0=3e-4, weight_decay=1e-4, cos_lr=True, close_mosaic=2, workers=4,
                seed=SEED + 1, deterministic=True, project=str(WORK / "runs"), name=NAME, exist_ok=True,
                verbose=False)
    shutil.copy(WORK / "runs" / NAME / "weights" / "last.pt", weights)
    found = weights
print("weights:", found)

# %%
log_file = WORK / "runs" / NAME / "results.csv"
log = pd.read_csv(log_file).rename(columns=str.strip) if log_file.exists() else None
log[["epoch", "time", "metrics/mAP50-95(B)"]].round(4) if log is not None else "no training log (weights reused)"

# %% [markdown]
# ## Test detections: variant `all_nms`
# Each view is saved, so other fusions can be tried later on CPU.

# %%
model = load_ultralytics(found)


def detector(imgsz):
    def detect(images):  # RGB arrays -> (xyxy, cls, score) per image
        res = model.predict([cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images], imgsz=imgsz, conf=0.001,
                            max_det=300, half=True, verbose=False)
        return [(b.xyxy.reshape(-1, 4), b.cls.astype(int), b.conf) for b in (r.boxes.cpu().numpy() for r in res)]
    return detect


VIEW_SPECS = {"full": (1280, "full"), "flip": (1280, "flip"), "full1600": (1600, "full"), "tiles": (1280, "tiles")}
VARIANT = dict(views=["full", "flip", "full1600", "tiles"], iou=0.65)

test = pd.read_csv(DATA / "test_images.csv")
test_paths = [str(DATA / f) for f in test["file_name"]]
test_views = {}
for v, (imgsz, view) in VIEW_SPECS.items():
    f = WORK / f"test_view_{NAME}_{v}.parquet"
    test_views[v] = pd.read_parquet(f) if f.exists() else collect_view(detector(imgsz), test_paths, view)
    test_views[v].to_parquet(f)
r2 = variant_dets(test_views, VARIANT)
r2.to_parquet(WORK / f"test_dets_{NAME}.parquet")
write_submission(r2, test, WORK / f"submission_{NAME}.csv").head()

# %% [markdown]
# ## Fusion with the earlier submissions
# Class-wise NMS (the merge that beat WBF on val when boxes come from different views). R1 and R2 are the same
# architecture, so their scores are comparable as they are. The YOLO ensemble has another score scale, so in the
# three-way fusion every source's scores are replaced by their percentile within that source first.
#
# Submit `submission_r1_r2.csv` first; `submission_r1_r2_yolo.csv` only if that one beats 0.3293.

# %%
def find_input(name):
    return next(iter(sorted(INPUT.rglob(name))), None) if INPUT.exists() else None


def read_submission(path):
    """Kaggle submission -> dets (key, cls, score, x, y, w, h), keys as in `image_key`."""
    sub = pd.read_csv(path, keep_default_na=False)
    key_of = dict(zip(test["image_id"], test["file_name"].map(image_key)))
    parts = []
    for image_id, s in zip(sub["image_id"], sub["PredictionString"]):
        g = np.array(s.split(), dtype=float).reshape(-1, 6)
        parts.append(pd.DataFrame({"key": key_of[image_id], "cls": g[:, 0].astype(int), "score": g[:, 1],
                                   "x": g[:, 2], "y": g[:, 3], "w": g[:, 4], "h": g[:, 5]}))
    return pd.concat(parts, ignore_index=True)


def ranked(dets):
    return dets.assign(score=dets["score"].rank(pct=True).to_numpy())


r1_file = find_input("test_dets_rtdetr_tiles.parquet")
r1 = pd.read_parquet(r1_file) if r1_file else None
if r1 is None and find_input("submission_rtdetr_tiles.csv"):
    r1 = read_submission(find_input("submission_rtdetr_tiles.csv"))
yolo_file = find_input("submission_final.csv")
yolo = read_submission(yolo_file) if yolo_file else None
print("R1:", r1_file or "csv" if r1 is not None else "not attached", "| YOLO ensemble:", yolo_file or "not attached")

if r1 is not None:
    write_submission(merge_views([r1, r2], FUSE_IOU), test, WORK / "submission_r1_r2.csv")
    print("written: submission_r1_r2.csv")
if r1 is not None and yolo is not None:
    write_submission(merge_views([ranked(r1), ranked(r2), ranked(yolo)], FUSE_IOU), test,
                     WORK / "submission_r1_r2_yolo.csv")
    print("written: submission_r1_r2_yolo.csv")
if yolo is not None:
    write_submission(merge_views([ranked(r2), ranked(yolo)], FUSE_IOU), test, WORK / "submission_r2_yolo.csv")
    print("written: submission_r2_yolo.csv")
