# %% [markdown]
# # Custom architecture: which detection levels does VisDrone need?
#
# YOLO11s predicts on three FPN levels: P3 (stride 8), P4 (16), P5 (32). The EDA showed 24% of objects under
# 32×32 px and few large ones, so we ask which levels are worth their cost. Four variants, all under one
# protocol (YOLO11s, COCO weights, masked frames, 1280 px, every 5th frame, 20 epochs, last epoch, fold 0):
#
# | Variant | Levels | Change vs YOLO11s |
# |---|---|---|
# | `yolo_masked` | P3, P4, P5 | — (reference) |
# | `yolo_p2` | P2, P3, P4, P5 | + stride-4 level for tiny objects |
# | `yolo_p2_nop5` | P2, P3, P4 | + P2, − P5 |
# | `yolo_nop5` | P3, P4 | − P5 |
#
# Hypotheses: (1) P2 helps small objects, (2) P5 (objects > ~256 px) is rare in VisDrone, so removing it costs
# little AP_large and saves compute. The variants train two at a time, one per T4 (~5 h in total).
#
# Kaggle: GPU T4 x2, Internet on, competition data attached. A finished run attached as input is reused, and
# `baseline` (02, no masking) is added to the table if its output is attached.

# %%
HOURS = 4.0         # safety cap per variant; the 20 epochs normally finish earlier
EPOCHS = 20         # upper bound, as in 07_0
WORKERS = 2         # dataloader workers per run (two runs share 4 CPUs)

# %% [markdown]
# ## Setup

# %%
import subprocess
import sys
from pathlib import Path

if Path("../src/core").exists():
    SRC = Path("../src").resolve()
else:
    repo = Path("/tmp/repo")  # outside /kaggle/working so it is not saved as output
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/mohinska/ai_hw2_MultipleObjectVideoDetection", str(repo)], check=True)
    SRC = repo / "src"
sys.path.insert(0, str(SRC))

# %%
# !pip install -q ultralytics faster-coco-eval

# %%
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from core import CLASSES
from core.cv import compare, existing, load_oof
from core.data import assign_folds, every_nth, find_data_root, load_pool, pool_targets
from core.errors import match_detections, outcome_tables
from core.yolo import P2_CFG, load_ultralytics, plot_curves, write_masked

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../outputs/arch")
WORK.mkdir(parents=True, exist_ok=True)
print(torch.cuda.device_count(), "GPU(s)")

# %% [markdown]
# ## Architectures
# The new heads are the YOLO11s head (and the P2 head of 07_0) with the P5 branch removed: no downsampling
# step from P4 to P5 in the head, and `Detect` on the remaining levels. The backbone is unchanged, so the COCO
# weights of every backbone layer still load.

# %%
NOP5_YAML = """# YOLO11s, detection on P3 and P4 only
nc: 8
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]] # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16)
  - [[16, 19], 1, Detect, [nc]]
"""

p2 = P2_CFG.read_text().replace("\r\n", "\n")
P2_NOP5_YAML = p2[:p2.index("  - [-1, 1, Conv, [512, 3, 2]]\n  - [[-1, 10]")] + "  - [[19, 22, 25], 1, Detect, [nc]]\n"
assert "[[19, 22, 25], 1, Detect" in P2_NOP5_YAML and "# 28" not in P2_NOP5_YAML

ARCH = Path("/tmp/arch")
ARCH.mkdir(exist_ok=True)
(ARCH / "yolo11s-nop5.yaml").write_text(NOP5_YAML)
(ARCH / "yolo11s-p2-nop5.yaml").write_text(P2_NOP5_YAML)

BASE = dict(model="yolo11s.pt", imgsz=1280, epochs=EPOCHS, batch=8, train_step=5, val_step=10,
            conf=0.001, iou=0.7, max_det=500, mask_ignore=True)
CONFIGS = {
    "yolo_masked": dict(BASE),
    "yolo_p2": {**BASE, "model_cfg": str(P2_CFG), "batch": 4},  # batch 4 so the extra level fits a T4
    "yolo_p2_nop5": {**BASE, "model_cfg": str(ARCH / "yolo11s-p2-nop5.yaml"), "batch": 4},
    "yolo_nop5": {**BASE, "model_cfg": str(ARCH / "yolo11s-nop5.yaml")},
}
for name, cfg in CONFIGS.items():
    cfg["name"] = name
    cfg["train_args"] = {"time": HOURS, "workers": WORKERS}
VARIANTS = list(CONFIGS)
QUEUES = [["yolo_p2", "yolo_nop5"], ["yolo_p2_nop5", "yolo_masked"]]  # GPU 0, GPU 1: a slow and a fast run each

# %% [markdown]
# Size and compute of each architecture (built from the yaml, before training).

# %%
YAMLS = {"yolo_masked": "yolo11s.yaml", "yolo_p2": str(P2_CFG),
         "yolo_p2_nop5": str(ARCH / "yolo11s-p2-nop5.yaml"), "yolo_nop5": str(ARCH / "yolo11s-nop5.yaml")}
from ultralytics.utils.torch_utils import get_flops

size = {}
for name, y in YAMLS.items():
    m = load_ultralytics(y)
    params = sum(p.numel() for p in m.model.parameters())
    size[name] = {"params_M": params / 1e6, "GFLOPs_1280": get_flops(m.model, imgsz=1280),
                  "strides": m.model.stride.int().tolist()}
size = pd.DataFrame(size).T
size.round(2)

# %% [markdown]
# ## Data
# Fold 0 of the shared folds. The masked training copies are written once here, so the two parallel runs only
# read them.

# %%
DATA = find_data_root()
frames, gt = load_pool(DATA)
targets = pool_targets(gt)
frames["fold"] = frames["video"].map(assign_folds(frames))
held0 = frames[frames["fold"] == 0]
train0 = every_nth(frames[frames["fold"] != 0], BASE["train_step"])
write_masked(train0, targets, Path("/tmp/masked_frames"))
print(len(train0), "training frames,", len(held0), "held-out frames")

# %% [markdown]
# ## Training: two queues, one per GPU
# Each GPU runs one process that sees only that GPU and trains its variants one after the other with the same
# `run_cv(..., folds=[0])` as the model notebooks. A finished run attached as input is reused.

# %%
RUNNER = '''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from core.cv import run_cv
from core.data import find_data_root, load_pool, pool_targets
from core.yolo import make_yolo_fns
frames, gt = load_pool(find_data_root())
targets = pool_targets(gt)
for cfg, work in json.loads(sys.argv[1]):
    print("start", cfg["name"], flush=True)
    work = Path(work)
    run_cv(frames, gt, cfg, *make_yolo_fns(cfg, work, targets), work, folds=[0])
'''
(ARCH / "runner.py").write_text(RUNNER)

def run_dir(name):
    return Path(f"/tmp/run_{name}")


procs = {}
for gpu, queue in enumerate(QUEUES):
    todo = [n for n in queue if not existing(f"metrics_{n}_fold0.json", WORK)]
    print(f"GPU {gpu}:", todo or "all done")
    if not todo:
        continue
    for n in todo:
        run_dir(n).mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(WORK / f"log_gpu{gpu}.txt", "w")
    jobs = json.dumps([[CONFIGS[n], str(run_dir(n))] for n in todo])
    procs[gpu] = subprocess.Popen([sys.executable, str(ARCH / "runner.py"), jobs, "-", str(SRC)],
                                  env=env, stdout=log, stderr=subprocess.STDOUT)
    time.sleep(180)  # the first run writes Ultralytics' label cache of the shared masked frames; the second reuses it

start = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    print(f"{(time.time() - start) / 60:.0f} min:", {f"GPU {g}": "running" if p.poll() is None else p.returncode
                                                   for g, p in procs.items()})

for gpu, p in procs.items():
    print(f"GPU {gpu} exit code", p.returncode)
    lines = (WORK / f"log_gpu{gpu}.txt").read_text().splitlines()
    print("\n".join(l for l in lines if l.startswith(("start", "fold", "  mAP", "Traceback")) or "Error" in l))
for name in VARIANTS:
    d = run_dir(name)
    for f in [f"{name}_fold0.pt", f"oof_{name}_fold0.parquet", f"metrics_{name}_fold0.json", f"cv_{name}.json"]:
        if (d / f).exists():
            shutil.copy(d / f, WORK / f)
    if (d / "runs").exists():
        shutil.copytree(d / "runs", WORK / "runs", dirs_exist_ok=True)

# %% [markdown]
# ## Results on fold 0
# `yolo_masked` is the reference (same data, standard head). Metrics on all 6,819 held-out frames.

# %%
table = compare([*VARIANTS, "baseline"], WORK, fold=0).join(size)
table.round(4)

# %%
epochs = {}
for name in VARIANTS:
    csv = WORK / "runs" / f"{name}_fold0" / "results.csv"
    hits = [csv] + sorted(Path("/kaggle/input").rglob(f"{name}_fold0/results.csv"))
    hit = next((h for h in hits if h.exists()), None)
    epochs[name] = len(pd.read_csv(hit)) if hit else None
print("epochs trained:", epochs)
for name in VARIANTS:
    plot_curves(name, 1, WORK);

# %% [markdown]
# ### Per-class AP
# Which classes each level matters for (bus and truck are the largest objects, so they should feel P5 most).

# %%
per_class = {}
for name in VARIANTS:
    f = existing(f"metrics_{name}_fold0.json", WORK)
    if f:
        m = json.loads(f.read_text())
        per_class[name] = {k[3:]: v for k, v in m.items() if k[3:] in CLASSES}
pd.DataFrame(per_class).round(3)

# %% [markdown]
# ## Failure analysis
# Every object of every 5th held-out frame is labelled found / missed / wrong class / poor box (detections with
# score >= 0.3, IoU >= 0.5), and every detection correct / duplicate / background. Shares by object size show
# where each level change wins or loses.

# %%
sample = every_nth(held0, 5)
missed, det_out = {}, {}
for name in VARIANTS:
    if not existing(f"oof_{name}_fold0.parquet", WORK):
        print(name, "no OOF detections")
        continue
    objects, detections = match_detections(load_oof(name, [0], WORK), sample, gt)
    by_size, _, det_share = outcome_tables(objects, detections)
    missed[name] = by_size.get("missed")
    det_out[name] = det_share
print("share of objects missed, by size:")
display(pd.DataFrame(missed).round(3))
print("detections (score >= 0.3) by outcome:")
pd.DataFrame(det_out).round(3)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
pd.DataFrame(missed).plot.bar(ax=ax)
ax.set(ylabel="share of objects missed", title="Missed objects by size, fold 0")
plt.tight_layout()
fig.savefig(WORK / "arch_missed_by_size.png", dpi=120)
