# %% [markdown]
# # Boost: one stronger model on folds 1-3, test-time augmentation chosen on fold 0
#
# One run = one model trained on folds 1, 2, 3 (held-out fold 0 is the same as in 07_1 / 07_2), so its fold-0
# detections can be compared and fused with the existing models without looking at the leaderboard.
# - **E1:** YOLO11m, 1280 px, every 3rd frame.
# - **E2:** YOLO11s + P2 head, 1600 px, every 3rd frame.
#
# Training time is capped by `HOURS` (Ultralytics fits the LR schedule into it). Then the fold-0 frames and the
# test frames are predicted with every view (two scales, flip); the views are saved, so the ensemble
# notebook can recombine them on CPU.
#
# Kaggle: GPU T4 x2, Internet on, competition data attached. Set `EXP` below.

# %%
EXP = "e1"      # "e1" or "e2"
HOURS = 4.5     # training budget in hours (the whole notebook takes ~1.2 h more)
DEVICE = [0, 1]  # both T4s (DDP); use 0 if DDP fails

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
import pandas as pd
import torch

from core.cv import cv_table, environment, run_cv
from core.data import assign_folds, every_nth, find_data_root, load_pool, pool_targets
from core.submission import write_submission
from core.variants import choose_variant, collect_view, compare_variants, variant_dets
from core.yolo import P2_CFG, load_ultralytics, make_yolo_fns, plot_curves

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../outputs/boost")
WORK.mkdir(parents=True, exist_ok=True)
print(torch.cuda.device_count(), "GPU(s)")
environment()

# %% [markdown]
# ## Config

# %%
COMMON = dict(epochs=40, train_step=3, val_step=20, conf=0.001, iou=0.7, max_det=500, mask_ignore=True)
AUG = dict(close_mosaic=3, hsv_v=0.5, scale=0.6)  # brighter/darker and smaller copies: test is darker, 8 clips 680-960 px
CONFIGS = {
    "e1": dict(name="boost_y11m_1280", model="yolo11m.pt", imgsz=1280, batch=8, scales=[1280, 1600],
               train_args=dict(**AUG)),
    # pretrained= so that the P2 model also gets COCO weights inside the DDP subprocess (it rebuilds from the yaml)
    "e2": dict(name="boost_y11s_p2_1600", model="yolo11s.pt", model_cfg=str(P2_CFG), imgsz=1600, batch=4,
               scales=[1600, 1920], train_args=dict(**AUG, pretrained="yolo11s.pt")),
}
CFG = {**COMMON, **CONFIGS[EXP]}
CFG["train_args"] = {**CFG["train_args"], "time": HOURS, "device": DEVICE, "workers": 4}
NAME = CFG["name"]
CFG

# %% [markdown]
# ## Data and folds

# %%
DATA = find_data_root()
frames, gt = load_pool(DATA)
targets = pool_targets(gt)
frames["fold"] = frames["video"].map(assign_folds(frames))
held0 = every_nth(frames[frames["fold"] == 0], 5)  # same held-out frames as 07_1 / 07_2
print(len(held0), "held-out frames of fold 0,", held0["video"].nunique(), "clips")

# %% [markdown]
# ## Train on folds 1-3, predict every fold-0 frame
# Model kept: `last.pt`. A finished run attached as input is reused.

# %%
result = run_cv(frames, gt, CFG, *make_yolo_fns(CFG, WORK, targets), WORK, folds=[0])
cv_table(result).round(4)

# %%
plot_curves(NAME, 1, WORK);

# %% [markdown]
# ## Test-time views on fold 0
# Views: full frame at both scales, flip at the training scale. Everything also on frames shrunk to 960 px
# (low-resolution test clips). A variant is kept only if it beats the single full pass at both resolutions and in
# most clips (`choose_variant`, as in 07_1).

# %%
weights = WORK / f"{NAME}_fold0.pt"
if not weights.exists():
    weights = next(Path("/kaggle/input").rglob(f"{NAME}_fold0.pt"))
model = load_ultralytics(weights)


def detector(imgsz):
    def detect(images):  # RGB arrays -> (xyxy, cls, score) per image
        res = model.predict([cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images], imgsz=imgsz, conf=0.001,
                            iou=0.7, max_det=500, half=True, verbose=False)
        return [(b.xyxy.reshape(-1, 4), b.cls.astype(int), b.conf) for b in (r.boxes.cpu().numpy() for r in res)]
    return detect


S1, S2 = CFG["scales"]
VIEW_SPECS = {f"full{S1}": (S1, "full"), f"flip{S1}": (S1, "flip"), f"full{S2}": (S2, "full")}
F1, P1, F2 = VIEW_SPECS
REF = "full"
VARIANTS = {
    REF: dict(views=[F1]),
    "flip_nms": dict(views=[F1, P1], iou=0.65),
    "flip_wbf": dict(views=[F1, P1], iou=0.65, fuse="wbf"),
    "scales_nms": dict(views=[F1, F2], iou=0.65),
    "scales_wbf": dict(views=[F1, F2], iou=0.65, fuse="wbf"),
    "scales_flip_nms": dict(views=[F1, P1, F2], iou=0.65),
    "scales_flip_wbf": dict(views=[F1, P1, F2], iou=0.65, fuse="wbf"),
}


def collect_all(paths, max_side=None):
    return {v: collect_view(detector(s), paths, view, max_side=max_side) for v, (s, view) in VIEW_SPECS.items()}


paths = held0["path"].tolist()
native_views, low_views = collect_all(paths), collect_all(paths, max_side=960)
for v in VIEW_SPECS:
    native_views[v].to_parquet(WORK / f"fold0_view_{NAME}_{v}.parquet")
    low_views[v].to_parquet(WORK / f"fold0_view_{NAME}_{v}_lowres.parquet")

native = compare_variants(native_views, VARIANTS, held0, gt, REF)
lowres = compare_variants(low_views, VARIANTS, held0, gt, REF, clips=False)
CHOSEN = choose_variant(native, lowres, REF)
print("chosen:", CHOSEN)
tta = native.join(lowres[["mAP"]].rename(columns={"mAP": "mAP_960"}))
tta.to_csv(WORK / f"tta_{NAME}.csv")
# same file names as 07_1, so the ensemble notebook treats this model like the others
variant_dets(native_views, VARIANTS[CHOSEN]).to_parquet(WORK / f"fold0_dets_{NAME}.parquet")
variant_dets(low_views, VARIANTS[CHOSEN]).to_parquet(WORK / f"fold0_dets_{NAME}_lowres.parquet")
(WORK / f"chosen_variant_{NAME}.json").write_text(json.dumps(
    {"name": NAME, "chosen": CHOSEN, "variants": VARIANTS, "views": VIEW_SPECS}, indent=1))
tta

# %% [markdown]
# ## Test set
# All views are saved (`test_view_*`); the chosen variant gives a single-model submission for a leaderboard check.

# %%
test = pd.read_csv(DATA / "test_images.csv")
test_paths = [str(DATA / f) for f in test["file_name"]]
test_views = collect_all(test_paths)
for v, d in test_views.items():
    d.to_parquet(WORK / f"test_view_{NAME}_{v}.parquet")
test_dets = variant_dets(test_views, VARIANTS[CHOSEN])
test_dets.to_parquet(WORK / f"test_dets_{NAME}.parquet")
write_submission(test_dets, test, WORK / f"submission_{NAME}.csv").head()
