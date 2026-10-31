# Multiple Object Video Detection

Our solution to Homework 2 of the "Artificial Intelligence" course at UCU: detect objects in drone video frames from [VisDrone2019-VID](https://arxiv.org/abs/2001.06303).

## Task

For every frame of 48 unlabelled test clips, predict boxes, classes and scores. Boxes are in original-image pixels, at most 500 per image.

- **Classes:** person, bicycle, car, van, truck, tricycle, bus, motor.
- **Metrics:** COCO mAP, AP50, AP_small and inference speed on val.
- **Ignored regions:** crowds and "others" are not penalised. They are masked in training and marked `iscrowd=1` in evaluation.

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

Train and val are split by clip, never by frame. Details: `notebooks/01_eda_preprocessing.ipynb`.

## Plan

Every detector is trained under one fixed protocol (`src/vod/protocol.py`): same data, resolution, epochs and evaluation.

| Step | Model | Notebook |
|---|---|---|
| 1. Data understanding | class balance, object sizes, density, ignored regions | `01_eda_preprocessing` |
| 2. Baseline | YOLO11s, Ultralytics defaults | `02_baseline_yolo` |
| 3. Two-stage anchor-based | Faster R-CNN: RPN proposals, RoI sampling, box loss, failure cases | `03_faster_rcnn` |
| 4. One-stage anchor-based | RetinaNet: anchors from data, IoU thresholds, focal loss | `04_retinanet` |
| 5. One-stage anchor-free | FCOS: tuning, breakdown by object size | — |
| 6. Query-based | RT-DETR: queries, losses, same compute budget | — |
| 7. Final pipeline | best model plus data and inference strategies, submission | — |

## Repository

```
notebooks/    one notebook per step above
src/vod/      data loading, models and training, COCO evaluation, error analysis, submission
report.md
images/
```
