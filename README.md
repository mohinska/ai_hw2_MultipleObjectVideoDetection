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

Train and val are pooled (63 clips, 55 source flights) and split into **4 folds with `GroupKFold`, grouped by source flight**: clips cut from one flight (`uav0000013_00000_v`, `uav0000013_01073_v`) are always in the same fold, so nothing leaks between frames of one video.

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
report.md
images/
```

### `src/core`

| Module | What it does |
|---|---|
| `__init__.py` | `CLASSES`, `SEED`, `N_FOLDS`, and `PROTOCOL`: the fixed settings of the detector comparison (03-06) |
| `data.py` | `load_pool` (train + val frames and COCO ground truth with unique ids), `assign_folds` (grouped folds), `every_nth` (frame step), `pool_targets` (boxes and ignored regions per frame), `load_coco` (one split as DataFrames), `find_data_root` |
| `evaluate.py` | `coco_evaluate`: COCO mAP / AP50 / AP_small / per-class AP, ignored regions as `iscrowd=1`, up to 500 detections per image; `per_clip_map` |
| `cv.py` | `run_cv`: the cross-validation loop for any model, resumable from saved folds; `folds=[0]` for sweeps; `cv_table`, `compare` (runs side by side), `load_oof`, `best_per_family`, `environment` (versions + GPU) |
| `tv.py` | torchvision Faster R-CNN, RetinaNet and FCOS: `FrameDataset` (resize, ignored-region masking, flip), `make_tv_fns` for `run_cv`, batch-1 speed, `two_stage_breakdown`, `proposal_sweep`, `anchor_match_stats`, `fcos_assignment_stats`, `plot_histories` |
| `errors.py` | failure analysis: `match_detections` (missed / wrong class / poor box / duplicate / background), `outcome_tables`, `worst_frames`, `plot_failures` |
| `yolo.py` | `make_yolo_fns`: Ultralytics (YOLO and RT-DETR) training and prediction plugged into `run_cv` (optional masked training copies via `write_masked`, P2 head via `P2_CFG`); `plot_curves` and `plot_budget_curves` (mAP vs training hours) from `results.csv` |
| `fusion.py` | `wbf` (Weighted Boxes Fusion, optional per-model weights) and `fold_ensemble`: every fold model predicts the test set, boxes are fused |
| `variants.py` | inference variants of a fold model: `make_detector`, `collect_view` (full / flip / tiles, optional low resolution), `merge_views` (NMS or WBF), `compare_variants`, `choose_variant` |
| `temporal.py` | `rescore`: a box confirmed in neighbouring frames of the clip keeps its score, a box seen once loses part of it |
| `viz.py` | `frcnn_gradcam` (ResNet stages and FPN levels), `frcnn_roi_embeddings`, `cam_concentration`, `rtdetr_probe` and `attention_stats` (deformable attention), `wandb_log_run` (curves, config, predictions with boxes) |
| `submission.py` | `write_submission`: Kaggle CSV plus format checks (ids, ≤ 500 boxes, class range, pixel coordinates) |

A model family plugs into the CV loop with two functions:

```python
train_fn(train_frames, held_out_frames, tag) -> weights_path
predict_fn(weights_path, image_paths) -> (detections[key, cls, score, x, y, w, h], ms_per_image)
result = run_cv(frames, gt, cfg, train_fn, predict_fn, work_dir)
```