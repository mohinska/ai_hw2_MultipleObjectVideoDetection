"""Training and inference for torchvision detectors (Faster R-CNN, RetinaNet, FCOS) under one protocol."""

import json
import math
import random
import shutil
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
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou, sigmoid_focal_loss

from . import NUM_CLASSES
from .evaluate import coco_evaluate, per_clip_map


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

        img, scale = resize_long_side(img, self.cfg["imgsz"])
        boxes *= scale

        if self.train and self.bank is not None and random.random() < self.cfg.get("paste_p", 0.0):
            img, boxes, cls = paste_objects(img, boxes, cls, self.bank, self.cfg)
        if self.train:
            img, boxes = augment(img, boxes, self.cfg)

        keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        target = {"boxes": torch.from_numpy(boxes[keep]), "labels": torch.from_numpy(cls[keep] + 1)}
        meta = {"path": row["path"], "scale": scale}
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255), target, meta


def resize_long_side(img, size):
    """Image with its long side equal to `size`, and the scale that was applied."""
    scale = size / max(img.shape[:2])
    # INTER_AREA is best for shrinking but acts like nearest-neighbour when enlarging (low-res test clips)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=interp), scale


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
                              score_thresh=cfg.get("score_thresh", 0.001),
                              center_sampling_radius=cfg.get("center_sampling_radius", 1.5))
        # replace only the last conv: the pretrained 4-conv tower (and the box/centerness branches) are kept
        head = m.head.classification_head
        head.cls_logits = torch.nn.Conv2d(256, n, 3, padding=1)
        torch.nn.init.normal_(head.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(head.cls_logits.bias, -math.log(99))  # prior 0.01, as in the focal-loss paper
        head.num_classes = n
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
            # optional loss re-weighting, e.g. {"loss_box_reg": 2.0} for the Faster R-CNN box loss
            weights = cfg.get("loss_weights", {})
            loss = sum(v * weights.get(k, 1.0) for k, v in losses.items())
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


def evaluate_run(model, ds, gt_json, cfg, out_dir, name, weights="last.pt"):
    """Final evaluation on `ds`: COCO metrics (+ per class, per clip) and batch-1 speed;
    saved as metrics.json and val_dets.parquet."""
    out_dir = Path(out_dir)
    dets, _ = predict(model, ds, cfg)
    metrics = coco_evaluate(gt_json, dets, file_names=ds.frames["path"], per_class=True)
    metrics.update(name=name, weights=weights, ms_per_image=round(measure_speed(model, ds), 1),
                   per_clip_mAP=per_clip_map(gt_json, dets, ds.frames["path"]),
                   config={k: v for k, v in cfg.items() if k != "seed"})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    dets.to_parquet(out_dir / "val_dets.parquet")
    return metrics, dets


@torch.no_grad()
def measure_speed(model, ds, n=100, warmup=10):
    """Median forward time per image at batch 1 (image loading excluded), the same way for every model."""
    device = _device()
    model.to(device).eval()
    times = []
    for i in range(min(n + warmup, len(ds))):
        img = ds[i][0].to(device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            model([img])
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def evaluate_checkpoint(name, cfg, ckpt, val_frames, val_targets, val_gt, out_dir, history=None):
    """Evaluate a saved checkpoint on every val frame; `history` (a history.csv) is copied next to it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if history is not None and Path(history).resolve() != (out_dir / "history.csv").resolve():
        shutil.copy(history, out_dir / "history.csv")
    model = build_model({**cfg, "pretrained": False})
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    metrics, _ = evaluate_run(model, FrameDataset(val_frames, val_targets, cfg), val_gt, cfg, out_dir, name,
                              weights=Path(ckpt).name)
    del model
    torch.cuda.empty_cache()
    return metrics


def run_experiment(name, cfg, train_frames, train_targets, val_frames, val_targets, val_gt, out_dir):
    """Train under `cfg`, then evaluate the last epoch on every val frame. The best epoch is chosen
    on val, so reporting it would be optimistic; best.pt is kept only for reference."""
    out_dir = Path(out_dir)
    seed_everything(cfg["seed"])
    model = build_model(cfg)
    train(model, FrameDataset(train_frames, train_targets, cfg, train=True),
          FrameDataset(val_frames.iloc[::cfg["val_step_epoch"]], val_targets, cfg), val_gt, cfg, out_dir)
    del model
    torch.cuda.empty_cache()
    return evaluate_checkpoint(name, cfg, out_dir / "last.pt", val_frames, val_targets, val_gt, out_dir)


def completed_runs(roots=(Path("/kaggle/working"), Path("/kaggle/input"))):
    """metrics.json of every finished run under `roots`, keyed by run name (first root wins)."""
    runs = {}
    for root in map(Path, roots):
        for p in sorted(root.rglob("metrics.json")) if root.exists() else []:
            m = json.loads(p.read_text(encoding="utf-8-sig"))
            if m["name"] in runs:
                continue
            m["dir"] = p.parent
            hist = p.parent / "history.csv"
            m["history"] = pd.read_csv(hist) if hist.exists() else None
            runs[m["name"]] = m
    return runs


def run_all(names, experiments, protocol, train_frames, train_targets, val_frames, val_targets, val_gt, work):
    """Train what has not been trained; re-evaluate runs trained earlier but evaluated with older code
    (best.pt, no per-clip mAP) from their attached last.pt. Every finished run is saved immediately."""
    for name in names:
        cfg = {**protocol, **experiments[name]}
        done = completed_runs((work, Path("/kaggle/input")))
        if name in done and done[name].get("weights") == "last.pt" and "per_clip_mAP" in done[name]:
            print(f"{name}: done")
            continue
        if name in done and (done[name]["dir"] / "last.pt").exists():
            print(f"{name}: re-evaluating last.pt")
            m = evaluate_checkpoint(name, cfg, done[name]["dir"] / "last.pt", val_frames, val_targets, val_gt,
                                    work / "runs" / name, history=done[name]["dir"] / "history.csv")
        else:
            print(f"{name}: training {experiments[name]}")
            m = run_experiment(name, cfg, train_frames, train_targets, val_frames, val_targets, val_gt,
                               work / "runs" / name)
        print({k: round(m[k], 4) for k in ["mAP", "AP50", "AP75", "AP_small"]}, m["ms_per_image"], "ms/image")


def comparison_tables(runs, classes, names=None):
    """Main metrics, per-class AP and per-clip mAP of the selected runs."""
    sel = {n: m for n, m in runs.items() if names is None or n in names}
    cols = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500", "ms_per_image"]
    main = pd.DataFrame({n: {c: m.get(c) for c in cols} for n, m in sel.items()}).T.astype(float).round(4)
    per_class = pd.DataFrame({n: {c: m.get(f"AP_{c}") for c in classes} for n, m in sel.items()}).T.round(4)
    per_clip = pd.DataFrame({n: m.get("per_clip_mAP", {}) for n, m in sel.items()}).T
    return main, per_class, per_clip


def load_run_model(name, cfg, roots=(Path("/kaggle/working"), Path("/kaggle/input"))):
    """Model of a finished run from its last.pt, wherever it was saved or attached."""
    ckpt = next(p for root in map(Path, roots) if root.exists() for p in root.rglob(f"{name}/last.pt"))
    model = build_model({**cfg, "pretrained": False})
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def predict_test(model, cfg, data_root):
    """Detections for every test frame, keyed by the submission image_id."""
    test = pd.read_csv(Path(data_root) / "test_images.csv")
    frames = pd.DataFrame({"path": [str(Path(data_root) / f) for f in test["file_name"]]})
    dets, _ = predict(model, FrameDataset(frames, {}, cfg), cfg)
    dets["image_id"] = dets.pop("file_name").map(dict(zip(frames["path"], test["image_id"])))
    return dets


# ---------------------------------------------------------------- two-stage analysis

@torch.no_grad()
def proposal_sweep(model, ds, val_gt, cfg, counts=(100, 300, 1000, 2000)):
    """Same Faster R-CNN weights, different numbers of RPN proposals kept at test time."""
    pre, post = dict(model.rpn._pre_nms_top_n), dict(model.rpn._post_nms_top_n)
    rows = []
    for n in counts:
        model.rpn._pre_nms_top_n["testing"] = max(n, pre["testing"])
        model.rpn._post_nms_top_n["testing"] = n
        dets, _ = predict(model, ds, cfg)
        m = coco_evaluate(val_gt, dets, file_names=ds.frames["path"])
        rows.append({"proposals": n, **{k: round(m[k], 4) for k in ["mAP", "AP50", "AP_small", "AR500"]},
                     "ms_per_image": round(measure_speed(model, ds, n=50), 1)})
    model.rpn._pre_nms_top_n.update(pre)
    model.rpn._post_nms_top_n.update(post)
    return pd.DataFrame(rows)


@torch.no_grad()
def two_stage_breakdown(model, ds, n_frames=300, top_n=(100, 300, 1000), score_thr=0.05, seed=0):
    """What each Faster R-CNN stage contributes, by object size at network input.

    RPN: share of GT boxes covered (IoU >= 0.5) by the top-n proposals, and the best proposal IoU.
    Second stage: share covered by final detections (score >= score_thr) and their best IoU;
    the IoU gain is the box refinement, classification is measured by AP.
    """
    device = _device()
    model.to(device).eval()
    captured = {}
    hooks = [model.rpn.register_forward_hook(lambda m, i, o: captured.__setitem__("proposals", o[0])),
             model.transform.register_forward_hook(lambda m, i, o: captured.__setitem__("sizes", o[0].image_sizes))]
    rows = []
    for i in random.Random(seed).sample(range(len(ds)), min(n_frames, len(ds))):
        img, target, _ = ds[i]
        gt = target["boxes"]
        if not len(gt):
            continue
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model([img.to(device)])[0]
        th, tw = captured["sizes"][0]
        props = captured["proposals"][0].float().cpu() * torch.tensor(
            [img.shape[2] / tw, img.shape[1] / th, img.shape[2] / tw, img.shape[1] / th])
        dets = out["boxes"][out["scores"] >= score_thr].float().cpu()
        row = {"side": ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).sqrt().numpy()}
        for n in top_n:
            row[f"rpn_recall@{n}"] = (box_iou(gt, props[:n]).max(1).values >= 0.5).numpy() if len(props) else 0
        row["rpn_best_iou"] = box_iou(gt, props).max(1).values.numpy() if len(props) else 0.0
        row["final_best_iou"] = box_iou(gt, dets).max(1).values.numpy() if len(dets) else 0.0
        row["final_recall"] = row["final_best_iou"] >= 0.5
        rows.append(pd.DataFrame(row))
    for h in hooks:
        h.remove()
    df = pd.concat(rows, ignore_index=True)
    df["size"] = pd.cut(df["side"], [0, 16, 32, 64, np.inf], labels=["<16", "16-32", "32-64", ">64"])
    recall_cols = [c for c in df.columns if "recall" in c]
    table = df.groupby("size", observed=True).agg(
        objects=("side", "size"), **{c: (c, "mean") for c in recall_cols},
        rpn_median_iou=("rpn_best_iou", "median"), final_median_iou=("final_best_iou", "median"))
    table.index = table.index.astype(str)
    table.loc["all"] = [len(df), *[df[c].mean() for c in recall_cols],
                        df["rpn_best_iou"].median(), df["final_best_iou"].median()]
    return table.round(3)


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


def fcos_positive_locations(boxes, stride, radius):
    """FCOS locations (grid points k * stride) that are positive for each xyxy box.

    A location is positive if it lies inside the box and within `radius * stride` of the box centre
    (torchvision's centre sampling; ignores the per-level size range and overlaps between boxes).
    """
    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    lo = np.maximum(boxes[:, :2], centre - radius * stride)
    hi = np.minimum(boxes[:, 2:], centre + radius * stride)
    per_axis = np.maximum(np.floor(hi / stride) - np.ceil(lo / stride) + 1, 0)
    return (per_axis[:, 0] * per_axis[:, 1]).astype(int)


def fcos_assignment_stats(frames, targets, imgsz, radius, stride=8):
    """Positive locations per training box at the finest FCOS level, by object size at network input."""
    rows = []
    for _, r in frames.iterrows():
        boxes = targets[r["id"]][0] * (imgsz / max(r["width"], r["height"]))
        if len(boxes):
            rows.append(pd.DataFrame({"side": np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])),
                                      "positives": fcos_positive_locations(boxes, stride, radius)}))
    df = pd.concat(rows, ignore_index=True)
    df["size"] = pd.cut(df["side"], [0, 16, 32, 64, np.inf], labels=["<16", "16-32", "32-64", ">64"])
    table = df.groupby("size", observed=True).agg(
        objects=("side", "size"), no_positive=("positives", lambda p: (p == 0).mean()),
        median_positives=("positives", "median"))
    table.index = table.index.astype(str)
    table.loc["all"] = [len(df), (df["positives"] == 0).mean(), df["positives"].median()]
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
            # sigmoid heads (RetinaNet, FCOS) also score label 0, which is background and never a target
            keep = out["labels"] > 0
            out = {k: v[keep] for k, v in out.items()}
            b = out["boxes"].float().cpu().numpy() / meta["scale"]
            part = pd.DataFrame({"x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]},
                                dtype="float32")
            part["cls"] = (out["labels"].cpu().numpy() - 1).astype("int8")
            part["score"] = out["scores"].float().cpu().numpy()
            part["file_name"] = meta["path"]
            parts.append(part)
    return pd.concat(parts, ignore_index=True), 1000 * seconds / max(n, 1)
