---
title: Homework 2 -- Multiple Object Video Detection
author: Maryna Ohinska, Mariia Hamaniuk
course: Artificial Intelligence · UCU
date: 2026-11-10
---

## Introduction

_TODO_

## Data

Code: `notebooks/01_eda.ipynb`.

| | train | val | test |
|---|---|---|---|
| clips / frames | 56 / 24,201 | 7 / 2,846 | 48 / 3,811 |
| objects (per frame) | 1,105,516 (45.7) | 114,132 (40.1) | — |
| ignored annotations | 510,744 | 31,960 | — |
| frame widths, px | 1344–2720 | 1344–3840 | 680–1920 |
| mean brightness (0–255) | 116 | 108 | 100 |

Data: No missing values, no zero-size, OOF boxes, duplicates, gaps in frame numbering, and the jpg count matches its annotations. 352 boxes have an aspect ratio above 6:1; they are objects cut by the frame border, so we keep them. 3 train frames have no objects.

Classes: Car and person dominate train; bus is 0.9% of objects. mAP averages over classes, so bus counts as much as car. Val has a different mix because it is only 7 clips. We do not rebalance val. Bus appears in 25 of 63 clips and in only one val clip, so val AP for bus is unreliable.

![Class distribution](images/eda_classes.png)

![Objects per clip for each class](images/eda_class_per_clip.png)

**Object sizes → input resolution and anchors.** By COCO area, 24% of objects are small, 64% medium, 12% large; person is 37% small, bus 57% large. The share of objects that shrink below 8 px (one cell of the finest stride) depends on the input size:

| imgsz | 640 | 1024 | 1280 | 1536 |
|---|---|---|---|---|
| objects below 8 px at network input | 9.7% | 1.4% | 0.6% | 0.2% |

We train at 1280: 640 loses about a tenth of the objects, and 1536 gains little for its cost. k-means on box shapes at 1280 gives clusters from 9×16 to 124×99 px with h/w between 0.5 and 1.9 (person ≈ 2.0, car ≈ 0.8). This sets the anchor sizes (16–256) and aspect ratios used for the anchor-based models.

![Object sizes](images/eda_object_sizes.png)

**Scene density.** Frames hold 42 objects at the median and 180 at most; 7.5% have more than 100. The per-clip median ranges from 5 to 150. We keep up to 500 detections per image and need at least 300 queries for query-based models.

**Test differs from train.** Eight test clips are 680–960 px wide, smaller than any train clip (≥1344 px). Upscaled to 1280, they are much blurrier: the median Laplacian variance at network input is 84–215 for test versus 497–848 for train. Test is also darker (mean brightness 100 vs 116; share of dark pixels 0.20 vs 0.11), and its darkest clip has a mean of 22. One test clip is a 2×2 collage of four videos. This motivates the downscale/JPEG and brightness augmentations that we test in the final pipeline.

![Clip brightness](images/eda_brightness_per_clip.png)

![Low-resolution test clips vs train at network input](images/eda_lowres_test.jpg)

**Ignored regions and non-target classes.** Train has 498,952 ignored regions (crowds, parking lots, distant people) and 11,792 "others" (mixers, excavators, carts), about 20 per frame. Their median side is 94 px versus 46 px for objects. 72% of frames contain them, and in 2,536 frames they cover more than half of the image. 2.7% of target objects lie at least half inside one. Val has only 8 "others".

![Ground truth with ignored regions](images/eda_gt_samples.jpg)

![Ignored regions and "others"](images/eda_ignore_crops.jpg)

**Temporal redundancy → frame step.** The median IoU of the same track between frames t and t+k is 0.90 (k=1), 0.73 (k=3), 0.59 (k=5) and 0.35 (k=10). Neighbouring frames are near-duplicates, so we train on every 5th frame (4,868 frames). That saves 5× GPU time while consecutive training frames still differ noticeably.

![Same-track IoU vs frame step](images/eda_track_iou.png)

## Preprocessing and validation

**Conversion.** Ultralytics models read the provided YOLO labels. For torchvision, `vod.data.load_split` converts the COCO json to pixel `xyxy` boxes with labels 1–8 (0 is background) and keeps the ignored regions separately. On 60 val frames, the YOLO and COCO labels agree exactly: same box counts, no class mismatches, 0 px difference. The figures above and below confirm the boxes visually.

**Ignored regions and "others".** During training, both are filled with the frame's mean colour, and target boxes that are at least 50% covered are dropped, so the model is neither taught that crowds are background nor asked to find objects on a grey patch. During evaluation, they are `iscrowd=1`, so detections inside them are not counted. The YOLO baseline (`02`) was trained without this masking.

**Splits.** Clip names show that several clips are segments of one source video (6 source videos have 2–3 clips). The provided val (7 clips, no source video shared with train) is the fixed hold-out for comparing detectors. For the final pipeline, we use 5-fold GroupKFold grouped by source video (11 sources and ~5,400 frames per fold), so no video is split across folds. Test clip names are anonymised, so overlap with train cannot be checked.

**Resizing, normalization, augmentation.** Frames are resized so that the long side is 1280 (area interpolation), and boxes are scaled with them. torchvision models then divide pixels by 255 and normalise with ImageNet mean/std; Ultralytics models divide by 255. All detector comparisons share one protocol (`src/vod/protocol.py`): every 5th train frame, horizontal flip (p=0.5), masking of ignored regions, 8 epochs, seed 42. Extra augmentations are kept out of the comparison and tested only in the final pipeline:

| augmentation | reason (from the data) |
|---|---|
| brightness/contrast jitter, p=0.5 | test is darker than train |
| downscale ×0.35–0.75 + JPEG q30–75, p=0.3 | low-resolution, blurry test clips |
| copy-paste of clean bicycle/van/truck/tricycle/bus objects, p=0.5 | rare classes; the object bank is built from **train only** (964 buses) |
| repeat-factor sampling, t=0.5 | rare classes; weak here: bus frames rise only from 27% to 34%, because bus frames are common while bus objects are not |

![YOLO labels vs. the torchvision training sample: ignored regions masked, rare object pasted](images/eda_preprocessing.jpg)

**Evaluation.** COCO mAP, AP50 and AP_small (faster-coco-eval) on every val frame, with up to 500 detections per image and ignored regions skipped.

## Two-stage anchor-based

_TODO_

## One-stage anchor-based

_TODO_

## One-stage anchor-free

_TODO_

## Query-based

_TODO_

## Final pipeline

_TODO_

## Results

| Model | mAP | AP50 | AP_small | ms / image | Public LB |
|---|---|---|---|---|---|
| YOLO11s baseline | | | | | |

## How to run

_TODO_
