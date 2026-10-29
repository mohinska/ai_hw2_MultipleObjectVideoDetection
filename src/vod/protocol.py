"""The fixed protocol every detector comparison uses: same data, resolution, budget and evaluation.

Experiments override only the model-specific keys; anything that changes the data or
augmentation belongs to the final-pipeline experiments, not to the detector comparison.
"""

from . import SEED

PROTOCOL = dict(
    imgsz=1280,          # long side; 0.6% of objects fall below 8 px here vs 9.7% at 640 (01_eda)
    min_size=720,
    train_step=5,        # every 5th frame: same-track IoU at step 5 is 0.59 (01_eda)
    val_step_epoch=10,   # val subset for the per-epoch curve; final numbers use every val frame
    epochs=8,
    batch=4,
    eval_batch=4,
    workers=4,
    seed=SEED,
    max_det=500,
    score_thresh=0.001,
    mask_ignore=True,
    hflip=0.5,
    color_p=0.0,
    degrade_p=0.0,
    paste_p=0.0,
)
