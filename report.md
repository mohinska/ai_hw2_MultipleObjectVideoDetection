---
title: Homework 2 -- Multiple Object Video Detection
author: Maryna Ohinska, Mariia Hamaniuk
course: Artificial Intelligence · UCU
date: 2026-11-10
---

## Introduction

_TODO_

## Data

| | train | val | test |
|---|---|---|---|
| clips / frames | 56 / 24,201 | 7 / 2,846 | 48 / 3,811 |
| objects (per frame) | 1,105,516 (45.7) | 114,132 (40.1) | — |
| ignored annotations | 510,744 | 31,960 | — |
| frame widths, px | 1344–2720 | 1344–3840 | 680–1920 |
| mean brightness (0–255) | 116 | 108 | 100 |

| imgsz | 640 | 1024 | 1280 | 1536 |
|---|---|---|---|---|
| objects below 8 px at network input | 9.7% | 1.4% | 0.6% | 0.2% |

_TODO: findings and the decisions they lead to._

![Class distribution](images/eda_classes.png)

![Objects per clip for each class](images/eda_class_per_clip.png)

![Object sizes](images/eda_object_sizes.png)

![Ground truth with ignored regions](images/eda_gt_samples.jpg)

![Ignored regions and "others"](images/eda_ignore_crops.jpg)

![Clip brightness](images/eda_brightness_per_clip.png)

![Low-resolution test clips vs train at network input](images/eda_lowres_test.jpg)

![Same-track IoU vs frame step](images/eda_track_iou.png)

## Preprocessing and validation

_TODO_

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
