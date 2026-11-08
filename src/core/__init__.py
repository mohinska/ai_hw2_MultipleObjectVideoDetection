"""Shared tools for the VisDrone-VID notebooks: data pool and grouped folds, COCO evaluation,
the cross-validation loop and the Kaggle submission. Model-specific code lives in its own module (core.yolo)."""

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
SEED = 42
N_FOLDS = 4
MAX_DETS = 500  # per-image cap of the leaderboard
