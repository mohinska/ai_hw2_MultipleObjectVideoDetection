---
title: Homework 2 - Multiple Object Video Detection
author: Maryna Ohinska, Mariia Hamaniuk
course: Artificial Intelligence · UCU
date: 2026-11-10
---

## Introduction

We detect 8 object classes (person, bicycle, car, van, truck, tricycle, bus, motor) in drone video frames from VisDrone2019-VID and submit boxes for 48 unlabelled test clips. The labelled data has **56 train clips (24,201 frames)** and **7 val clips (2,846 frames)** with 1.2 million objects.

The data analysis showed four main difficulties:

- **Small objects:** 24% of objects are smaller than 32×32 px (COCO "small"), so the input resolution decides how many objects are visible at all.
- **Dense scenes:** 42 objects per frame at the median and up to 180, so duplicate suppression and the detection limit matter.
- **Class imbalance:** car and person are three quarters of the objects, bus is under 1%, while mAP weighs every class equally.
- **Correlated frames and a shift to test:** neighbouring frames are near-duplicates, clips come from shared source flights, and the test set has darker and lower-resolution clips.

We compare four detector families under one protocol and out-of-fold validation, next to a YOLO11s baseline, and then build the final pipeline from data, inference and ensemble strategies on top of the best models. Code: [`src/core`](src/core/), one notebook per step in [`notebooks/`](notebooks/).

## Method

### Data preparation

Code: [01_eda_preprocessing.ipynb](notebooks/01_eda_preprocessing.ipynb).

| | train | val | test |
|---|---|---|---|
| clips / frames | 56 / 24,201 | 7 / 2,846 | 48 / 3,811 |
| objects (per frame) | 1,105,516 (45.7) | 114,132 (40.1) | — |
| frame widths, px | 1344–2720 | 1344–3840 | 680–1920 |

The annotations are clean: no zero-size or out-of-frame boxes, no duplicates, no gaps in frame numbering. Each finding sets one design choice:

| Finding | Decision |
|---|---|
| 24% of objects are small; at 640 px, 9.7% shrink below 8 px (the finest stride), at 1280 px only 0.6%. | Train and predict at **1280 px**. |
| k-means box shapes at 1280 px: 9×16 to 124×99 px, h/w 0.5–1.9. | Test anchors **16–256 px** instead of 32–512. |
| Median 42 objects per frame, up to 180. | Up to **500 detections** per image, **300 queries** for RT-DETR. |
| Same-track IoU between frames t and t+k: 0.90 (k=1), 0.59 (k=5), 0.35 (k=10). | Train on **every 5th frame**: 5× less compute, frames still differ. |
| 20 ignored regions and "others" per frame, in 72% of frames. | Mask them in training, skip them in evaluation. |
| 8 test clips are 680–960 px wide; test is darker (mean brightness 100 vs 116). | Measure the loss at 960 px in the final pipeline. |

![Object sizes](images/eda_object_sizes.png)

Ultralytics reads the YOLO txt labels and torchvision the COCO json; on 60 val frames both agree exactly. Ignored regions and "others" are `iscrowd=1` in evaluation, so detections inside them do not count. In training they are filled with the frame's mean colour and objects at least 50% inside them are dropped, so crowds are not taught as background. Only the baseline trains without this masking.

![Ground truth with ignored regions](images/eda_gt_samples.jpg)

### Validation strategy

The official val split has 7 clips, and bus appears in only one of them, so a single holdout is a noisy benchmark. Several clips are also segments of one source flight. We pool train and val (63 clips, 55 flights, 27,047 frames) and split them into **4 folds with `GroupKFold` by flight** ([`assign_folds`](src/core/data.py)), so no flight is split across folds.

Each fold model predicts every frame of its held-out fold, and models are compared by the **out-of-fold (OOF) mAP** over all labelled frames. We keep the last epoch, since picking the best one on the held-out fold would inflate OOF. k-means anchors are fitted on the training folds only. Metrics are COCO mAP, AP50, AP_small and per-class AP ([`coco_evaluate`](src/core/evaluate.py)) with up to 500 detections per image, plus batch-1 time per image on a T4.

For the test set, the 4 fold models each predict and their boxes are merged with Weighted Boxes Fusion ([`fold_ensemble`](src/core/fusion.py)). This gain cannot be scored OOF, so it is measured on the public leaderboard.

### Model selection and tuning

All detectors share one protocol ([`PROTOCOL`](src/core/__init__.py)): COCO-pretrained weights, long side 1280 px, every 5th training frame, horizontal flip, ignored-region masking, 8 epochs, batch 4, seed 42. Each family changes only its own parameters, one at a time, on fold 0; the best experiment gets the full 4-fold CV (its fold-0 score is slightly optimistic). Data and inference changes are kept for the final pipeline, so that their effect is measured separately from the model.

| Family | Model | Experiments (one change each) | Notebook |
|---|---|---|---|
| Baseline | YOLO11s, Ultralytics defaults | — | [02](notebooks/02_baseline.ipynb) |
| Two-stage anchor-based | Faster R-CNN v2, ResNet-50 FPN | RoI positive fraction 0.5; anchors 16–256; box loss ×2; step LR | [03](notebooks/03_faster_rcnn.ipynb) |
| One-stage anchor-based | RetinaNet v2 | anchors 16–256; + IoU 0.4/0.3; focal γ=0 | [04](notebooks/04_retinanet.ipynb) |
| Anchor-free | FCOS | centre radius 0.5 / 2.5; box loss ×2 | [05](notebooks/05_fcos.ipynb) |
| Query-based | RT-DETR-L, 1.5 GPU-h per fold | lr 3·10⁻⁴; no denoising queries | [06](notebooks/06_rtdetr.ipynb) |

## Results

### Baseline

YOLO11s, Ultralytics defaults (SGD, lr 0.01, mosaic), 20 epochs, 1280 px, every 5th frame, no ignore masking ([02_baseline.ipynb](notebooks/02_baseline.ipynb), [cv_baseline.json](reports/baseline/cv_baseline.json)).

| | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | ms / image |
|---|---|---|---|---|---|---|---|
| fold 0 | 0.195 | 0.367 | 0.179 | 0.039 | 0.148 | 0.337 | 35.6 |
| fold 1 | 0.247 | 0.450 | 0.236 | 0.067 | 0.254 | 0.429 | 35.5 |
| fold 2 | 0.156 | 0.321 | 0.131 | 0.050 | 0.167 | 0.313 | 35.7 |
| fold 3 | 0.203 | 0.363 | 0.199 | 0.069 | 0.178 | 0.362 | 35.8 |
| mean ± std | 0.200 ± 0.038 | 0.375 ± 0.054 | | 0.056 ± 0.014 | | | 35.6 |
| **OOF (all 27,047 frames)** | **0.188** | **0.356** | **0.173** | **0.053** | **0.175** | **0.334** | |

OOF AP per class: car 0.452, bus 0.281, person 0.201, tricycle 0.186, van 0.135, motor 0.123, bicycle 0.062, truck 0.060. Speed includes preprocessing and NMS; training took 1.5 h per fold on a T4.

![Baseline training curves per fold](images/baseline_curves.png)

The best epoch scores 0.011–0.026 mAP above the last one in every fold: this is how much `best.pt` would have inflated the OOF score. Fold mAP ranges from 0.156 to 0.247 depending on the held-out flights, so a single split could not separate most variants. Small objects are the main weakness (AP_small 0.053 vs AP_large 0.334); truck AP varies from 0.017 to 0.183 across folds because its few clips dominate.

On the public leaderboard, a single fold model scored 0.173; the WBF fusion of all 4 fold models scores **0.213**, since each fold model sees only 3/4 of the flights.

### Two-stage anchor-based: Faster R-CNN

SGD (momentum 0.9), lr 0.005, linear warmup and cosine decay, AMP. We measure what each stage contributes (RPN recall vs the second stage's detections) and split the failure cases into missed objects, wrong class, poor box, duplicates and background ([03_faster_rcnn.ipynb](notebooks/03_faster_rcnn.ipynb), [reports/faster_rcnn](reports/faster_rcnn/)).

| Experiment (fold 0) | mAP | AP50 | AP75 | AP_small | AP_large |
|---|---:|---:|---:|---:|---:|
| default | 0.225 | 0.435 | 0.202 | 0.039 | 0.390 |
| RoI positive fraction 0.5 | 0.228 | 0.441 | 0.209 | 0.039 | 0.395 |
| anchors 16–256 | 0.228 | 0.453 | 0.200 | 0.042 | 0.385 |
| box loss ×2 | 0.227 | 0.436 | 0.208 | 0.036 | 0.400 |
| **step LR** | **0.235** | 0.447 | 0.217 | 0.042 | 0.394 |

All four changes stay within 0.010 mAP of the default, which is smaller than the spread between folds. Step LR is the best on fold 0 and gets the full CV:

| step LR | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | ms / image |
|---|---|---|---|---|---|---|---|
| fold 0 | 0.235 | 0.447 | 0.217 | 0.042 | 0.209 | 0.394 | 66.3 |
| fold 1 | 0.247 | 0.466 | 0.230 | 0.058 | 0.238 | 0.405 | 64.9 |
| fold 2 | 0.185 | 0.387 | 0.150 | 0.063 | 0.188 | 0.407 | 68.4 |
| fold 3 | 0.229 | 0.432 | 0.218 | 0.067 | 0.208 | 0.426 | 65.7 |
| mean ± std | 0.224 ± 0.027 | 0.433 ± 0.034 | | 0.058 ± 0.011 | | | 66.3 |
| **OOF** | **0.212** | **0.416** | **0.190** | **0.055** | **0.193** | **0.401** | |

OOF AP per class: car 0.423, bus 0.356, person 0.212, tricycle 0.205, van 0.181, motor 0.131, truck 0.108, bicycle 0.079. Compared with the baseline, the OOF mAP is 0.024 higher, mostly from AP50 (+0.060) and large objects (+0.067); AP_small is about the same. Truck and bus gain the most. The model is 1.9× slower than YOLO11s. Public LB of the 4-fold WBF ensemble: **0.243** (baseline 0.213).

![Faster R-CNN held-out mAP per epoch, fold 0](images/frcnn_curves.png)

Held-out mAP peaks at epoch 2 and then oscillates within ±0.01, so 8 epochs at this learning rate mostly do not improve the model further.

![Faster R-CNN failure cases](images/frcnn_failure_cases.png)

The three frames with the most missed objects, wrong-class objects and background false positives. Misses are mostly tiny, distant objects and tightly packed parked cars; class errors are among the small two-wheelers and tricycles at a crowded crossing; false positives are parked cars far from the camera, which are probably unlabelled.

### One-stage anchor-based: RetinaNet

lr 0.0025. Without a proposal stage, the classifier scores every anchor, almost all of them background, and a small object may match no anchor at all; the focal loss and the anchor-matching statistics address these two issues. Anchors 16–256 come from the EDA box sizes ([04_retinanet.ipynb](notebooks/04_retinanet.ipynb), [reports/retinanet](reports/retinanet/)).

| Experiment (fold 0) | mAP | AP50 | AP75 | AP_small | AP_large |
|---|---:|---:|---:|---:|---:|
| default | 0.215 | 0.415 | 0.200 | 0.025 | 0.397 |
| anchors 16–256 | 0.199 | 0.418 | 0.165 | 0.042 | 0.328 |
| + IoU 0.4 / 0.3 | 0.192 | 0.408 | 0.155 | 0.041 | 0.320 |
| **focal γ = 0** | **0.221** | 0.419 | 0.208 | 0.025 | 0.400 |

Smaller anchors raise AP_small from 0.025 to 0.042 but cost 0.07 AP_large and 0.035 AP75, so the overall mAP falls; lower IoU thresholds make it worse. Removing the focal term (γ = 0) is slightly better than the default on fold 0. With the classification prior and 8 epochs, plain cross-entropy is enough here.

| focal γ = 0 | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | ms / image |
|---|---|---|---|---|---|---|---|
| fold 0 | 0.221 | 0.419 | 0.208 | 0.025 | 0.174 | 0.400 | 48.8 |
| fold 1 | 0.221 | 0.407 | 0.211 | 0.041 | 0.210 | 0.413 | 48.5 |
| fold 2 | 0.155 | 0.309 | 0.139 | 0.045 | 0.153 | 0.363 | 48.6 |
| fold 3 | 0.202 | 0.366 | 0.200 | 0.054 | 0.180 | 0.388 | 47.9 |
| mean ± std | 0.200 ± 0.031 | 0.375 ± 0.050 | | 0.041 ± 0.012 | | | 48.5 |
| **OOF** | **0.187** | **0.357** | **0.176** | **0.041** | **0.164** | **0.376** | |

OOF AP per class: car 0.405, bus 0.328, person 0.184, tricycle 0.183, van 0.155, truck 0.096, motor 0.091, bicycle 0.057. RetinaNet matches the baseline OOF (0.187 vs 0.188) and is 0.025 below Faster R-CNN, with the weakest AP_small of the three: without proposals and box refinement, small objects suffer most. Public LB: **0.188**.

![RetinaNet held-out mAP per epoch, fold 0](images/retinanet_curves.png)

### Anchor-free: FCOS

lr 0.0025. A location is positive if it lies inside a box and within a radius of its centre, on the FPN level that matches the box size; the radius sets how many locations a small object gets. Search range: radius 0.5, 1.5 (default), 2.5, and box loss ×2.

_Results pending._

### Query-based: RT-DETR

AdamW, cosine schedule, 300 queries. The compute budget is the baseline's: 1.5 GPU-hours per fold, and Ultralytics fits the epochs into this time. Held-out mAP over training hours is plotted against the YOLO baseline with the same budget.

_In progress._

### Comparison of all models

OOF over all labelled frames for the selected experiment of each family; fold-0 mAP for the initial (default) configuration.

| Model | Initial, fold-0 mAP | Selected experiment | OOF mAP | AP50 | AP_small | ms / image | Public LB |
|---|---:|---|---:|---:|---:|---:|---:|
| YOLO11s baseline | 0.195 | — | 0.188 | 0.356 | 0.053 | 35.6 | 0.213 |
| Faster R-CNN | 0.225 | step LR | 0.212 | 0.416 | 0.055 | 66.3 | 0.243 |
| RetinaNet | 0.215 | focal γ = 0 | 0.187 | 0.357 | 0.041 | 48.5 | 0.188 |
| FCOS | | | | | | | |
| RT-DETR-L | | | | | | | |

### Final pipeline

Three steps on fold 0, so that the effect of data and inference is separated from the effect of the model:

1. **Data and model** ([07_0_yolo.ipynb](notebooks/07_0_yolo.ipynb)): YOLO11s on masked data (the data effect alone), then a P2 high-resolution head and a higher classification loss weight.
2. **Inference** ([07_1_inference.ipynb](notebooks/07_1_inference.ipynb)): for the best run of each family, horizontal flip and 2×2 overlapping tiles merged by NMS or WBF, also at 960 px to mimic the low-resolution test clips.
3. **Ensemble** ([07_2_ensemble.ipynb](notebooks/07_2_ensemble.ipynb)): cross-model WBF of the top models, and temporal rescoring: a box confirmed in neighbouring frames keeps its score, a box seen once loses part of it ([`rescore`](src/core/temporal.py)).

_Results pending._

| Submission | Internal mAP | Public LB |
|---|---:|---:|
| YOLO11s baseline, single fold model | | 0.173 |
| YOLO11s baseline, fold ensemble (WBF of 4 fold models) | 0.188 (OOF) | 0.213 |
| Faster R-CNN step LR, fold ensemble | 0.212 (OOF) | 0.243 |
| RetinaNet γ = 0, fold ensemble | 0.187 (OOF) | 0.188 |
| Inference + temporal rescoring (07.1) | | 0.314 |
| Cross-model ensemble (07.2) | | 0.322 |

## Final model and reproducibility

### Final model

_Pending: the selected ensemble, its fold-0 / OOF score and public LB score._

### Reproducibility

Every notebook runs on Kaggle (GPU T4, Internet on) with the competition data attached. Its first cells clone this repository and install Ultralytics and faster-coco-eval; the package versions of our runs are in [requirements.txt](requirements.txt), and the seed is 42 everywhere.

1. Run the notebooks in order: 01 → 02 → 03, 04 → 05, 06, 07_0 → 07_1 → 07_2 (→ 08 for the visualizations).
2. Attach the outputs of the earlier notebooks as Kaggle inputs; a rerun with its own previous output attached resumes from the saved folds.
3. 07_2 writes the final `submission.csv`.

Every run saves `cv_<name>.json` with its config, seed, package versions and GPU.

## Conclusion

AI-usage: we used AI (Claude Opus 5.5) for generating code that is not connected to the final result itself (tables, plots e.g), to format some of our thoughts in the report, to correct grammar and punctuation both in notebooks and report.
## References

- Data: Zhu et al., *Detection and Tracking Meet Drones Challenge* (VisDrone), arXiv:2001.06303.
- Models: Ren et al., Faster R-CNN, arXiv:1506.01497; Lin et al., RetinaNet / focal loss, arXiv:1708.02002; Tian et al., FCOS, arXiv:1904.01355; Zhao et al., RT-DETR, arXiv:2304.08069; Ultralytics YOLO11 and RT-DETR (docs.ultralytics.com); torchvision detection models.
- Evaluation: COCO detection metrics (cocodataset.org/#detection-eval); faster-coco-eval (github.com/MiXaiLL76/faster_coco_eval).
- Ensembling: Solovyev et al., Weighted Boxes Fusion, arXiv:1910.13302.
- Visualization: Selvaraju et al., Grad-CAM, arXiv:1610.02391; Zhu et al., Deformable DETR, arXiv:2010.04159; van der Maaten & Hinton, t-SNE (JMLR 2008).
