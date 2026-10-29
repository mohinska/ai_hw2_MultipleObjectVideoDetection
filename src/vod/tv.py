"""Training and inference for torchvision detectors (Faster R-CNN, RetinaNet, FCOS) under one protocol."""

import json
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
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou, sigmoid_focal_loss

from . import NUM_CLASSES
from .evaluate import coco_evaluate


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


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
        # INTER_AREA is best for shrinking but acts like nearest-neighbour when enlarging (low-res test clips)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=interp)
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

RETINA_SIZES = (32, 64, 128, 256, 512)  # torchvision default base size per FPN level P3-P7


def retina_anchors(sizes, ratios):
    """RetinaNet anchors: 3 octave scales (x1, x2^1/3, x2^2/3) of each base size; ratios are h/w."""
    return AnchorGenerator(tuple(tuple(s * 2 ** (k / 3) for k in range(3)) for s in sizes),
                           (tuple(ratios),) * len(sizes))


class FocalClassificationHead(RetinaNetClassificationHead):
    """torchvision's RetinaNet classification head with configurable focal-loss alpha and gamma."""

    def __init__(self, in_channels, num_anchors, num_classes, alpha=0.25, gamma=2.0):
        super().__init__(in_channels, num_anchors, num_classes, norm_layer=partial(torch.nn.GroupNorm, 32))
        self.alpha, self.gamma = alpha, gamma

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        for tgt, logits, matched in zip(targets, head_outputs["cls_logits"], matched_idxs):
            fg = matched >= 0
            gt = torch.zeros_like(logits)
            gt[fg, tgt["labels"][matched[fg]]] = 1.0
            valid = matched != self.BETWEEN_THRESHOLDS
            losses.append(sigmoid_focal_loss(logits[valid], gt[valid], alpha=self.alpha, gamma=self.gamma,
                                             reduction="sum") / max(1, fg.sum()))
        return sum(losses) / len(targets)


def build_model(cfg):
    """Pretrained COCO detector with an 8-class head; torchvision label 0 is background."""
    n = NUM_CLASSES + 1
    pretrained = cfg.get("pretrained", True)
    common = dict(min_size=cfg["min_size"], max_size=cfg["imgsz"],
                  weights="DEFAULT" if pretrained else None, weights_backbone="DEFAULT" if pretrained else None)
    arch = cfg["arch"]

    if arch == "faster_rcnn":
        m = fasterrcnn_resnet50_fpn_v2(
            **common,
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
        m = retinanet_resnet50_fpn_v2(**common, detections_per_img=cfg.get("max_det", 500),
                                      score_thresh=cfg.get("score_thresh", 0.001),
                                      fg_iou_thresh=cfg.get("fg_iou", 0.5), bg_iou_thresh=cfg.get("bg_iou", 0.4))
        # 3 scales x 3 ratios per location as in the pretrained model, so the regression head still fits
        m.anchor_generator = retina_anchors(cfg.get("anchor_sizes", RETINA_SIZES), cfg.get("aspect_ratios", (0.5, 1.0, 2.0)))
        m.head.classification_head = FocalClassificationHead(
            256, m.anchor_generator.num_anchors_per_location()[0], n,
            alpha=cfg.get("focal_alpha", 0.25), gamma=cfg.get("focal_gamma", 2.0))
    elif arch == "fcos":
        m = fcos_resnet50_fpn(**common, detections_per_img=cfg.get("max_det", 500),
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
    device = _device()
    model.to(device)
    loader = make_loader(train_ds, cfg, shuffle=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("weight_decay", 1e-4))
    total, warmup = cfg["epochs"] * len(loader), min(500, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (it + 1) / warmup if it < warmup else
                                              0.5 * (1 + math.cos(math.pi * (it - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best = [], -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums, t0 = {}, time.time()
        for imgs, targets, _ in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
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


def evaluate_run(model, ds, gt_json, cfg, out_dir, name):
    """Final evaluation on `ds`: COCO metrics (+ per class), speed; saved as metrics.json and val_dets.parquet."""
    out_dir = Path(out_dir)
    dets, ms = predict(model, ds, cfg)
    metrics = coco_evaluate(gt_json, dets, file_names=ds.frames["path"], per_class=True)
    metrics.update(name=name, ms_per_image=round(ms, 1), config={k: v for k, v in cfg.items() if k != "seed"})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    dets.to_parquet(out_dir / "val_dets.parquet")
    return metrics, dets


@torch.no_grad()
def anchor_match_stats(frames, targets, sizes, ratios, imgsz, fg_iou, n_frames=200, seed=0):
    """How well an anchor set covers the training boxes, by object size at network input.

    For each box: best IoU with any anchor and the number of anchors with IoU >= fg_iou
    (boxes with none only get torchvision's single low-quality match).
    """
    gen = retina_anchors(sizes, ratios)
    cache, rows, per_image = {}, [], []
    for _, r in frames.sample(min(n_frames, len(frames)), random_state=seed).iterrows():
        scale = imgsz / max(r["width"], r["height"])
        h, w = math.ceil(r["height"] * scale / 32) * 32, math.ceil(r["width"] * scale / 32) * 32
        if (h, w) not in cache:
            feats = [torch.zeros(1, 1, math.ceil(h / s), math.ceil(w / s)) for s in (8, 16, 32, 64, 128)]
            cache[(h, w)] = gen(ImageList(torch.zeros(1, 3, h, w), [(h, w)]), feats)[0]
        anchors = cache[(h, w)]
        boxes = torch.from_numpy(targets[r["id"]][0] * scale)
        if not len(boxes):
            continue
        iou = box_iou(boxes, anchors)
        pos = iou >= fg_iou
        rows.append(pd.DataFrame({"side": ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sqrt().numpy(),
                                  "best_iou": iou.max(1).values.numpy(), "positives": pos.sum(1).numpy()}))
        per_image.append({"anchors": len(anchors), "positive_anchors": int(pos.any(0).sum())})
    df = pd.concat(rows, ignore_index=True)
    df["size"] = pd.cut(df["side"], [0, 16, 32, 64, np.inf], labels=["<16", "16-32", "32-64", ">64"])
    table = df.groupby("size", observed=True).agg(
        objects=("side", "size"), median_best_iou=("best_iou", "median"),
        no_positive_anchor=("positives", lambda p: (p == 0).mean()), median_positives=("positives", "median"))
    table.index = table.index.astype(str)
    img = pd.DataFrame(per_image).mean()
    table.loc["all"] = [len(df), df["best_iou"].median(), (df["positives"] == 0).mean(), df["positives"].median()]
    table.attrs["anchors_per_image"] = int(img["anchors"])
    table.attrs["negative_to_positive"] = round(img["anchors"] / max(img["positive_anchors"], 1))
    return table.round(3)


@torch.no_grad()
def predict(model, ds, cfg):
    """Detections in original-image pixels (file_name, cls 0-based, score, x, y, w, h) and ms per image."""
    device = _device()
    model.to(device).eval()
    parts, seconds, n = [], 0.0, 0
    for imgs, _, metas in make_loader(ds, cfg, shuffle=False):
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outs = model(imgs)
        _sync(device)
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
