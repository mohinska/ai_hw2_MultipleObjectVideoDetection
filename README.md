# Multiple Object Video Detection

Our solution to Homework 2 of the "Artificial Intelligence" course at UCU: detect objects in drone video frames from [VisDrone2019-VID](https://arxiv.org/abs/2001.06303).

## Task

For every frame of 48 unlabelled test clips, predict boxes, classes and scores. Boxes are in original-image pixels, at most 500 per image.

- **Classes:** person, bicycle, car, van, truck, tricycle, bus, motor.
- **Metrics:** COCO mAP, AP50, AP_small and inference speed on val.
- **Ignored regions:** crowds and "others" are not penalised. They are marked `iscrowd=1` in evaluation and masked in training (except the YOLO baseline, which trains on Ultralytics defaults).

## Team

| | Name | GitHub |
|:-:|---|---|
| <img src="https://github.com/mohinska.png?size=80" width="48" alt="mohinska"/> | Maryna Ohinska | [@mohinska](https://github.com/mohinska) |
| <img src="https://github.com/MariiaHamanuk.png?size=80" width="48" alt="MariiaHamanuk"/> | Mariia Hamaniuk | [@MariiaHamanuk](https://github.com/MariiaHamanuk) |

## Data

| Split | Clips | Frames | Labels |
|---|---:|---:|---|
| train | 56 | 24,201 | yes |
| val | 7 | 2,846 | yes |
| test | 48 | 3,811 | no |

Details: `notebooks/01_eda_preprocessing.ipynb`.

## Validation

Train and val are pooled (63 clips, 55 source flights) and split into **4 folds with `GroupKFold`, grouped by source flight**.

- For every fold a model is trained on the other folds (every 5th frame) and predicts every frame of the held-out fold. The out-of-fold (OOF) predictions cover each labelled frame once; **OOF mAP is the number models are compared by**.
- The model kept per fold is the last epoch (`last.pt`), so no checkpoint is picked by looking at the held-out fold.
- Learned preprocessing (e.g. k-means anchors) is fit on the training folds only.
- Each run saves `cv_<name>.json`: config, seed, package versions and GPU, per-fold and OOF metrics.

## Plan

Every detector is trained under one fixed protocol: same folds, resolution, epochs and evaluation.
All notebooks use the shared code in `src/core`.

| Step | Model | Notebook |
|---|---|---|
| 1. Data understanding | class balance, object sizes, density, ignored regions | [01_eda_preprocessing](notebooks/01_eda_preprocessing.ipynb) |
| 2. Baseline | YOLO11s, Ultralytics defaults | [02_baseline](notebooks/02_baseline.ipynb) |
| 3. Two-stage anchor-based | Faster R-CNN: RPN proposals, RoI sampling, box loss, failure cases | [03_faster_rcnn](notebooks/03_faster_rcnn.ipynb) |
| 4. One-stage anchor-based | RetinaNet: anchors from data, IoU thresholds, focal loss | [04_retinanet](notebooks/04_retinanet.ipynb) |
| 5. One-stage anchor-free | FCOS: centre-sampling radius, box loss, breakdown by object size | [05_fcos](notebooks/05_fcos.ipynb) |
| 6. Query-based | RT-DETR: queries, losses, same compute budget | [06_rtdetr](notebooks/06_rtdetr.ipynb) |
| 7.0 Pipeline: model | YOLO: masked data, P2 head, loss weight | [07_0_yolo](notebooks/07_0_yolo.ipynb) |
| 7.1 Pipeline: inference | flip, tiles, NMS vs WBF, robustness at 960 px | [07_1_inference](notebooks/07_1_inference.ipynb) |
| 7.2 Pipeline: ensemble | summary of all models, cross-model WBF, temporal rescoring, final submission | [07_2_ensemble](notebooks/07_2_ensemble.ipynb) |
| 8. Visualization (bonus) | W&B logs, Grad-CAM and RoI t-SNE of Faster R-CNN, RT-DETR deformable attention | [08_visualization](notebooks/08_visualization.ipynb) |

## Repository

```
notebooks/    one notebook per step above
src/core/     shared tools: data, folds, evaluation, CV loop, models, inference variants, ensembles, submission
reports/      saved CV results (config, metrics, training curves) of finished runs
images/       figures used in the report
report.md     the report; index.html renders it as a site
requirements.txt  package versions of the Kaggle runs
```

To view the report as a site locally (it loads `report.md`, so it needs a server, not `file://`):

```bash
python -m http.server 8000
# open http://localhost:8000
```

### `src/core`

**`__init__.py`**: `CLASSES`, `SEED`, `N_FOLDS`, `MAX_DETS` (500 per image) and `PROTOCOL`, the fixed training settings of the detector comparison (03–06).

**`data.py`**: loading data and folds
- `find_data_root`: finds the competition data on Kaggle or locally.
- `image_key`: `clip/frame` id of an image path.
- `load_coco`: one split (images, boxes) as DataFrames.
- `load_pool`: train + val frames and one COCO ground truth with unique ids.
- `assign_folds`: clip → fold, grouped by source flight.
- `every_nth`: keeps every n-th frame of each clip.
- `pool_targets`: boxes, classes and ignored regions per frame.
- `covered_share`: share of each box covered by ignored regions.

**`evaluate.py`**: metrics
- `coco_evaluate`: COCO mAP, AP50, AP_small, optional per-class AP; ignored regions count as `iscrowd=1`.
- `per_clip_map`: mAP of every clip separately.

**`cv.py`**: cross-validation
- `run_cv`: trains a model per fold, predicts the held-out fold, saves OOF predictions and metrics; skips folds already done.
- `environment`: package versions and GPU, saved with each run.
- `existing`: finds a saved file of this run or of an earlier run attached as Kaggle input.
- `cv_table`: per-fold metrics with mean ± std.
- `compare`: several runs side by side.
- `load_oof`: OOF predictions of a run.
- `best_per_family`: best run of each model family on fold 0.

**`tv.py`**: torchvision Faster R-CNN, RetinaNet, FCOS
- `FrameDataset`: reads a frame, resizes, masks ignored regions, flips.
- `build_model`, `load_model`: builds a detector from the config / loads trained weights.
- `retina_anchors`, `FocalClassificationHead`: custom anchors and the loss variant of RetinaNet.
- `train`, `lr_factor`: training loop with warmup and cosine / step LR schedule.
- `predict`, `measure_speed`: detections in original pixels; batch-1 time per image.
- `make_tv_fns`: train and predict functions for `run_cv`.
- `plot_histories`: training curves of several runs.
- `proposal_sweep`, `two_stage_breakdown`: Faster R-CNN, where objects are lost (RPN or second stage) and how many proposals are needed.
- `anchor_match_stats`: RetinaNet, how many objects get a positive anchor.
- `fcos_positive_locations`, `fcos_assignment_stats`: FCOS, which locations are positives for each object.

**`yolo.py`**: Ultralytics YOLO and RT-DETR
- `make_yolo_fns`: train and predict functions for `run_cv`.
- `write_masked`: training copies of frames with ignored regions greyed out.
- `load_ultralytics`: loads YOLO or RT-DETR weights.
- `plot_curves`, `plot_budget_curves`: mAP per epoch; mAP vs training hours.

**`errors.py`**: failure analysis
- `match_detections`: labels every object (found / missed) and detection (correct, wrong class, poor box, duplicate, background).
- `outcome_tables`: shares of these outcomes by class and size.
- `worst_frames`, `plot_failures`: frames with the most errors, drawn.

**`fusion.py`**: ensembles
- `iou_matrix`: pairwise IoU of two box sets.
- `wbf`: Weighted Boxes Fusion of several models' boxes.
- `fold_ensemble`: every fold model predicts the test set, boxes are fused with WBF.

**`variants.py`**: test-time inference variants
- `run_config`, `make_detector`: a fold model of a saved run as one detect function.
- `collect_view`: detections on full frames, flipped frames or tiles, optionally at low resolution.
- `merge_views`, `variant_dets`: combine views with NMS or WBF.
- `compare_variants`, `choose_variant`: mAP and speed of each variant; picks the one to use.

**`temporal.py`**: video
- `clip_neighbours`: neighbouring frames of each frame in its clip.
- `rescore`: a box also seen in neighbouring frames keeps its score, a box seen once loses part of it.

**`viz.py`**: visualization (08)
- `frcnn_gradcam`: Grad-CAM of one Faster R-CNN detection on ResNet stages and FPN levels.
- `cam_concentration`, `overlay`: how much of the map falls on the object; map drawn over the image.
- `frcnn_roi_embeddings`: 1024-d RoI features of boxes for t-SNE.
- `rtdetr_probe`, `attention_stats`: RT-DETR deformable-attention sampling points and weights; their distance to the box and share per feature level.
- `wandb_log_run`: logs a finished run to W&B (config, curves, metrics, predictions on images).

**`submission.py`**
- `write_submission`: Kaggle CSV from detections.
- `check_submission`: checks ids, ≤ 500 boxes, class range, pixel coordinates.
