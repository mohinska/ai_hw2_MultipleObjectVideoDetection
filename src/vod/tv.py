"""Training and inference for torchvision detectors (Faster R-CNN, RetinaNet, FCOS) under one protocol."""

import math
import random
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, fcos_resnet50_fpn,
                                          retinanet_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from . import NUM_CLASSES
from .evaluate import coco_evaluate


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data

class FrameDataset(Dataset):
    """Frames resized so the long side is `imgsz`; boxes are returned in the resized frame.

    train=True enables ignore-region masking and augmentation, as configured in `cfg`.
    `bank` (from vod.data.object_bank on the training split) enables copy-paste of rare-class objects;
    `weights` (from vod.data.repeat_factors) makes the loader oversample frames with rare classes.
    """

    def __init__(self, frames, targets, cfg, train=False, bank=None, weights=None):
        self.frames = frames.reset_index(drop=True)
        self.targets = targets
        self.cfg = cfg
        self.train = train
        self.bank = bank
        self.weights = weights

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        row = self.frames.iloc[i]
        img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
        boxes, cls, ignore = self.targets.get(row.get("id"), (np.zeros((0, 4), "float32"),
                                                              np.zeros(0, "int64"),
                                                              np.zeros((0, 4), "float32")))
        boxes, cls = boxes.copy(), cls.copy()

        if self.train and self.cfg.get("mask_ignore") and len(ignore):
            # grey out don't-care areas so they are not learned as background
            fill = img.reshape(-1, 3).mean(0).astype(np.uint8)
            for x1, y1, x2, y2 in ignore.astype(int):
                img[max(y1, 0):y2, max(x1, 0):x2] = fill
            # objects that are now mostly grey would teach the model to fire on the fill colour
            visible = 1 - covered_share(boxes, ignore)
            boxes, cls = boxes[visible >= 0.5], cls[visible >= 0.5]

        scale = self.cfg["imgsz"] / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        boxes *= scale

        if self.train and self.bank is not None and random.random() < self.cfg.get("paste_p", 0.0):
            img, boxes, cls = paste_objects(img, boxes, cls, self.bank, self.cfg)
        if self.train:
            img, boxes = augment(img, boxes, self.cfg)

        keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        target = {"boxes": torch.from_numpy(boxes[keep]), "labels": torch.from_numpy(cls[keep] + 1)}
        meta = {"path": row["path"], "scale": scale}
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255), target, meta


def covered_share(boxes, regions):
    """Share of each box's area covered by `regions` (overlapping regions counted twice, clipped to 1)."""
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes), "float32")
    ix = np.clip(np.minimum(boxes[:, None, 2], regions[None, :, 2]) - np.maximum(boxes[:, None, 0], regions[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, None, 3], regions[None, :, 3]) - np.maximum(boxes[:, None, 1], regions[None, :, 1]), 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.clip((ix * iy).sum(1) / np.maximum(area, 1e-6), 0, 1)


def paste_objects(img, boxes, cls, bank, cfg):
    """Paste 1..paste_max bank objects at free spots, scaled as they would be in their own frame."""
    h, w = img.shape[:2]
    for _ in range(random.randint(1, cfg.get("paste_max", 3))):
        o = bank.iloc[random.randrange(len(bank))]
        src = cv2.imread(o["path"])
        crop = cv2.cvtColor(src[int(o["y1"]):int(o["y2"]), int(o["x1"]):int(o["x2"])], cv2.COLOR_BGR2RGB)
        s = cfg["imgsz"] / o["long_side"]
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
        ch, cw = crop.shape[:2]
        if ch >= h or cw >= w:
            continue
        for _ in range(10):
            x, y = random.randrange(w - cw), random.randrange(h - ch)
            if len(boxes):
                ix = np.clip(np.minimum(x + cw, boxes[:, 2]) - np.maximum(x, boxes[:, 0]), 0, None)
                iy = np.clip(np.minimum(y + ch, boxes[:, 3]) - np.maximum(y, boxes[:, 1]), 0, None)
                if (ix * iy).max() > 0.05 * cw * ch:
                    continue
            img[y:y + ch, x:x + cw] = crop
            boxes = np.vstack([boxes, np.array([[x, y, x + cw, y + ch]], "float32")])
            cls = np.append(cls, o["cls"]).astype("int64")
            break
    return img, boxes, cls


def augment(img, boxes, cfg):
    h, w = img.shape[:2]
    if random.random() < cfg.get("hflip", 0.5):
        img = img[:, ::-1].copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if random.random() < cfg.get("color_p", 0.0):
        alpha = random.uniform(0.6, 1.4)          # contrast
        beta = random.uniform(-40, 40)            # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if random.random() < cfg.get("degrade_p", 0.0):
        # imitate the low-resolution, compressed test clips: downscale, upscale back, re-encode
        f = random.uniform(0.35, 0.75)
        small = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 75)])
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img, boxes


def collate(batch):
    return tuple(zip(*batch))


def make_loader(ds, cfg, shuffle):
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    sampler = None
    if shuffle and ds.weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(ds.weights, dtype=torch.double), len(ds),
                                        replacement=True, generator=g)
    return DataLoader(ds, batch_size=cfg["batch"] if shuffle else cfg.get("eval_batch", 4),
                      shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=cfg.get("workers", 4), collate_fn=collate, generator=g,
                      worker_init_fn=partial(_worker_seed, base=cfg["seed"]), persistent_workers=False)


def _worker_seed(worker_id, base):
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


# ---------------------------------------------------------------- models

def build_model(cfg):
    """Pretrained COCO detector with an 8-class head; torchvision label 0 is background."""
    n = NUM_CLASSES + 1
    common = dict(min_size=cfg["min_size"], max_size=cfg["imgsz"])
    arch = cfg["arch"]

    if arch == "faster_rcnn":
        m = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT", **common,
            rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_train", 2000),
            rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_train", 2000),
            rpn_pre_nms_top_n_test=cfg.get("rpn_pre_nms_test", 1000),
            rpn_post_nms_top_n_test=cfg.get("rpn_post_nms_test", 1000),
            box_batch_size_per_image=cfg.get("roi_batch", 512),
            box_positive_fraction=cfg.get("roi_pos_fraction", 0.25),
            box_detections_per_img=cfg.get("max_det", 500), box_score_thresh=cfg.get("score_thresh", 0.001),
        )
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n)
        if cfg.get("anchor_sizes"):
            # one size and three ratios per FPN level, so the pretrained RPN head still fits
            m.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in cfg["anchor_sizes"]),
                                                     ((0.5, 1.0, 2.0),) * len(cfg["anchor_sizes"]))
    elif arch == "retinanet":
        m = retinanet_resnet50_fpn_v2(weights="DEFAULT", **common, detections_per_img=cfg.get("max_det", 500),
                                      score_thresh=cfg.get("score_thresh", 0.001))
        if cfg.get("anchor_sizes"):
            m.anchor_generator = AnchorGenerator(
                tuple(tuple(s * 2 ** (k / 3) for k in range(3)) for s in cfg["anchor_sizes"]),
                (tuple(cfg.get("aspect_ratios", (0.5, 1.0, 2.0))),) * len(cfg["anchor_sizes"]))
        m.head.classification_head = RetinaNetClassificationHead(
            256, m.anchor_generator.num_anchors_per_location()[0], n)
    elif arch == "fcos":
        m = fcos_resnet50_fpn(weights="DEFAULT", **common, detections_per_img=cfg.get("max_det", 500),
                              score_thresh=cfg.get("score_thresh", 0.001))
        m.head.classification_head = FCOSClassificationHead(256, 1, n)
    else:
        raise ValueError(arch)
    return m


# ---------------------------------------------------------------- training

def train(model, train_ds, val_ds, val_gt, cfg, out_dir):
    """SGD + warmup + cosine schedule with AMP. Evaluates COCO mAP on `val_ds` after every epoch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model.to(device)
    loader = make_loader(train_ds, cfg, shuffle=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("weight_decay", 1e-4))
    total, warmup = cfg["epochs"] * len(loader), min(500, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (it + 1) / warmup if it < warmup else
                                              0.5 * (1 + math.cos(math.pi * (it - warmup) / (total - warmup))))
    scaler = torch.amp.GradScaler()

    history, best = [], -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums, t0 = {}, time.time()
        for imgs, targets, _ in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16):
                losses = model(imgs, targets)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()

        dets, ms = predict(model, val_ds, cfg)
        metrics = coco_evaluate(val_gt, dets, file_names=val_ds.frames["path"])
        row = {"epoch": epoch, "minutes": round((time.time() - t0) / 60, 1),
               **{k: round(v / len(loader), 4) for k, v in sums.items()},
               **{k: round(metrics[k], 4) for k in ["mAP", "AP50", "AP_small"]}}
        history.append(row)
        print(row, flush=True)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if metrics["mAP"] > best:
            best = metrics["mAP"]
            torch.save(model.state_dict(), out_dir / "best.pt")
    torch.save(model.state_dict(), out_dir / "last.pt")
    return pd.DataFrame(history)


@torch.no_grad()
def predict(model, ds, cfg):
    """Detections in original-image pixels (file_name, cls 0-based, score, x, y, w, h) and ms per image."""
    device = torch.device("cuda")
    model.to(device).eval()
    parts, seconds, n = [], 0.0, 0
    for imgs, _, metas in make_loader(ds, cfg, shuffle=False):
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            outs = model(imgs)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        n += len(imgs)
        for out, meta in zip(outs, metas):
            b = out["boxes"].float().cpu().numpy() / meta["scale"]
            part = pd.DataFrame({"x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]},
                                dtype="float32")
            part["cls"] = (out["labels"].cpu().numpy() - 1).astype("int8")
            part["score"] = out["scores"].float().cpu().numpy()
            part["file_name"] = meta["path"]
            parts.append(part)
    return pd.concat(parts, ignore_index=True), 1000 * seconds / max(n, 1)
