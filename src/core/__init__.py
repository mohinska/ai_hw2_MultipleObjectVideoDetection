"""Shared tools for the VisDrone-VID notebooks: data pool and grouped folds, COCO evaluation,
the cross-validation loop and the Kaggle submission. Model-specific code lives in its own module (core.yolo)."""

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
SEED = 42
N_FOLDS = 4
MAX_DETS = 500  # per-image cap of the leaderboard

# The fixed protocol of the detector comparison (03-06): same data, resolution, budget and evaluation.
# Experiments override only model-specific keys; data and augmentation changes belong to the final pipeline.
PROTOCOL = dict(
    imgsz=1280,          # long side; 0.6% of objects fall below 8 px here vs 9.7% at 640 (01_eda)
    min_size=720,        # short side for torchvision's transform (16:9 frames at long side 1280)
    train_step=5,        # every 5th frame: same-track IoU at step 5 is 0.59 (01_eda)
    val_step=10,         # held-out subset for the per-epoch curve; metrics use every held-out frame
    epochs=8,
    batch=4,
    eval_batch=4,
    workers=4,
    max_det=500,
    score_thresh=0.001,
    mask_ignore=True,    # grey out ignored regions, drop objects >= 50% inside them
    hflip=0.5,
)
