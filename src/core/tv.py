"""torchvision detectors (Faster R-CNN) as train_fn / predict_fn for core.cv.run_cv, plus two-stage analysis."""

import math
import random
import time
from functools import partial
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

from . import CLASSES, SEED
from .data import KAGGLE_INPUT, covered_share, every_nth, image_key
from .evaluate import coco_evaluate


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data

class FrameDataset(Dataset):
    """Frames resized so the long side is `imgsz`; boxes are returned in the resized frame.

    frames: DataFrame with `path` (and `uid` for labelled frames); targets: core.data.pool_targets.
    train=True greys out ignored regions (cfg mask_ignore) and applies horizontal flip (cfg hflip).
    """

    def __init__(self, frames, targets, cfg, train=False):
        self.frames = frames.reset_index(drop=True)
        self.targets = targets
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        row = self.frames.iloc[i]
        img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        empty = np.zeros((0, 4), "float32")
        boxes, cls, ignore = self.targets.get(row.get("uid"), (empty, np.zeros(0, "int64"), empty))
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
        if self.train and random.random() < self.cfg.get("hflip", 0.0):
            img = img[:, ::-1].copy()
            boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]

        keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        target = {"boxes": torch.from_numpy(boxes[keep]), "labels": torch.from_numpy(cls[keep] + 1)}
        meta = {"path": row["path"], "scale": scale, "size": (orig_w, orig_h)}
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255), target, meta


def resize_long_side(img, size):
    """Image with its long side equal to `size`, and the scale that was applied."""
    scale = size / max(img.shape[:2])
    # INTER_AREA is best for shrinking but acts like nearest-neighbour when enlarging (low-res test clips)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=interp), scale


def _collate(batch):
    return tuple(zip(*batch))


def _worker_seed(worker_id, base):
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


def make_loader(ds, cfg, shuffle):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=cfg["batch"] if shuffle else cfg.get("eval_batch", 4), shuffle=shuffle,
                      num_workers=cfg.get("workers", 4), collate_fn=_collate, generator=g,
                      worker_init_fn=partial(_worker_seed, base=SEED))


# ---------------------------------------------------------------- model

def build_model(cfg, pretrained=True):
    """COCO-pretrained Faster R-CNN (ResNet-50 FPN v2) with an 8-class head; label 0 is background."""
    w = "DEFAULT" if pretrained else None
    m = fasterrcnn_resnet50_fpn_v2(
        weights=w, weights_backbone=w, min_size=cfg["min_size"], max_size=cfg["imgsz"],
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_train", 2000),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_train", 2000),
        rpn_pre_nms_top_n_test=cfg.get("rpn_pre_nms_test", 1000),
        rpn_post_nms_top_n_test=cfg.get("rpn_post_nms_test", 1000),
        box_batch_size_per_image=cfg.get("roi_batch", 512),
        box_positive_fraction=cfg.get("roi_pos_fraction", 0.25),
        box_detections_per_img=cfg["max_det"], box_score_thresh=cfg["score_thresh"],
    )
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,
                                                  len(CLASSES) + 1)
    if cfg.get("anchor_sizes"):
        # one size and three ratios per FPN level, so the pretrained RPN head still fits
        m.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in cfg["anchor_sizes"]),
                                                 ((0.5, 1.0, 2.0),) * len(cfg["anchor_sizes"]))
    return m


def load_model(weights, cfg):
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model


# ---------------------------------------------------------------- train / predict

def lr_factor(it, total, warmup, schedule="cosine"):
    """Multiplier of the base lr at iteration `it`: linear warm-up, then
    cosine: smooth decay to 0; step: x0.1 at 2/3 and 8/9 of training (Detectron "1x" milestones)."""
    if it < warmup:
        return (it + 1) / warmup
    if schedule == "step":
        return 0.1 ** ((it >= total * 2 / 3) + (it >= total * 8 / 9))
    return 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(1, total - warmup)))


def train(model, train_ds, curve_frames, gt, cfg, history_path):
    """SGD + linear warmup + cosine schedule, AMP. After every epoch: mAP on `curve_frames` (a held-out
    subset), logged to `history_path` for the training curve only; the last epoch is the model kept."""
    device = _device()
    model.to(device)
    loader = make_loader(train_ds, cfg, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("weight_decay", 1e-4))
    total, warmup = cfg["epochs"] * len(loader), min(500, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_factor, total=total, warmup=warmup,
                                                           schedule=cfg.get("schedule", "cosine")))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_weights = cfg.get("loss_weights", {})  # e.g. {"loss_box_reg": 2.0}: loss modification experiment

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums, t0 = {}, time.time()
        for imgs, targets, _ in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                losses = model(imgs, targets)
            loss = sum(v * loss_weights.get(k, 1.0) for k, v in losses.items())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()

        dets = predict(model, curve_frames["path"].tolist(), cfg)
        m = coco_evaluate(dets, curve_frames, gt)
        history.append({"epoch": epoch, "minutes": round((time.time() - t0) / 60, 1),
                        **{k: round(v / len(loader), 4) for k, v in sums.items()},
                        **{k: round(m[k], 4) for k in ["mAP", "AP50", "AP_small"]}})
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"  epoch {epoch}: mAP {m['mAP']:.4f}, {history[-1]['minutes']} min", flush=True)
    return pd.DataFrame(history)


@torch.no_grad()
def predict(model, paths, cfg):
    """Detections in original-image pixels: key, cls (0-based), score, x, y, w, h."""
    device = _device()
    model.to(device).eval()
    parts = []
    for imgs, _, metas in make_loader(FrameDataset(pd.DataFrame({"path": paths}), {}, cfg), cfg, shuffle=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outs = model([im.to(device, non_blocking=True) for im in imgs])
        for out, meta in zip(outs, metas):
            # back to original pixels; the resized frame is rounded up to whole pixels, so clip to the original size
            w, h = meta["size"]
            b = np.clip(out["boxes"].float().cpu().numpy() / meta["scale"], 0, [w, h, w, h])
            parts.append(pd.DataFrame({
                "key": image_key(meta["path"]), "cls": (out["labels"].cpu().numpy() - 1).astype("int8"),
                "score": out["scores"].float().cpu().numpy(),
                "x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]}))
    return pd.concat(parts, ignore_index=True)


@torch.no_grad()
def measure_speed(model, paths, cfg, n=100, warmup=10):
    """Median forward time per image at batch 1 (image loading excluded), the same way for every model."""
    device = _device()
    model.to(device).eval()
    ds = FrameDataset(pd.DataFrame({"path": paths[:n + warmup]}), {}, cfg)
    warmup = min(warmup, len(ds) - 1)  # keep at least one timed image on tiny inputs
    times = []
    for i in range(len(ds)):
        img = ds[i][0].to(device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            model([img])
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def make_tv_fns(cfg, work, gt, targets):
    """train_fn / predict_fn for core.cv.run_cv. cfg: PROTOCOL plus model keys (lr, rpn_*, roi_*, anchor_sizes,
    loss_weights). Weights are the last epoch's state_dict saved as {tag}.pt; the curve goes to history_{tag}.csv."""
    work = Path(work)

    def train_fn(train_frames, held_out, tag):
        seed_everything()
        model = build_model(cfg)
        train(model, FrameDataset(train_frames, targets, cfg, train=True),
              every_nth(held_out, cfg["val_step"]), gt, cfg, work / f"history_{tag}.csv")
        weights = work / f"{tag}.pt"
        torch.save(model.state_dict(), weights)
        del model
        torch.cuda.empty_cache()
        return weights

    def predict_fn(weights, paths):
        model = load_model(weights, cfg)
        dets = predict(model, paths, cfg)
        ms = measure_speed(model, paths, cfg)
        del model
        torch.cuda.empty_cache()
        return dets, ms

    return train_fn, predict_fn


def plot_histories(names, fold, work):
    """Held-out mAP per epoch of several runs on one fold, from history_{name}_fold{k}.csv."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in names:
        hits = [Path(work) / f"history_{name}_fold{fold}.csv"]
        hits += sorted(KAGGLE_INPUT.rglob(f"history_{name}_fold{fold}.csv")) if KAGGLE_INPUT.exists() else []
        csv = next((h for h in hits if h.exists()), None)
        if csv is not None:
            h = pd.read_csv(csv)
            ax.plot(h["epoch"], h["mAP"], marker="o", label=name)
    ax.set(xlabel="epoch", ylabel="mAP (every 10th held-out frame)", title=f"Held-out mAP during training, fold {fold}")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(Path(work) / f"curves_fold{fold}.png", dpi=110)
    return fig


# ---------------------------------------------------------------- two-stage analysis

@torch.no_grad()
def proposal_sweep(model, frames, gt, cfg, counts=(100, 300, 1000, 2000)):
    """Same Faster R-CNN weights, different numbers of RPN proposals kept at test time."""
    pre, post = dict(model.rpn._pre_nms_top_n), dict(model.rpn._post_nms_top_n)
    paths = frames["path"].tolist()
    rows = []
    for n in counts:
        model.rpn._pre_nms_top_n["testing"] = max(n, pre["testing"])
        model.rpn._post_nms_top_n["testing"] = n
        m = coco_evaluate(predict(model, paths, cfg), frames, gt)
        rows.append({"proposals": n, **{k: round(m[k], 4) for k in ["mAP", "AP50", "AP_small", "AR500"]},
                     "ms_per_image": round(measure_speed(model, paths, cfg, n=50), 1)})
    model.rpn._pre_nms_top_n.update(pre)
    model.rpn._post_nms_top_n.update(post)
    return pd.DataFrame(rows)


@torch.no_grad()
def two_stage_breakdown(model, frames, targets, cfg, n_frames=300, top_n=(100, 300, 1000), score_thr=0.05):
    """What each Faster R-CNN stage contributes, by object size at network input.

    RPN: share of GT boxes covered (IoU >= 0.5) by the top-n proposals, and the best proposal IoU.
    Second stage: share covered by final detections (score >= score_thr) and their best IoU;
    the IoU gain is the box refinement, classification is measured by AP.
    """
    device = _device()
    model.to(device).eval()
    ds = FrameDataset(frames.sample(min(n_frames, len(frames)), random_state=SEED), targets, cfg)
    captured = {}
    hooks = [model.rpn.register_forward_hook(lambda m, i, o: captured.__setitem__("proposals", o[0])),
             model.transform.register_forward_hook(lambda m, i, o: captured.__setitem__("sizes", o[0].image_sizes))]
    rows = []
    for i in range(len(ds)):
        img, target, _ = ds[i]
        gt_boxes = target["boxes"]
        if not len(gt_boxes):
            continue
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model([img.to(device)])[0]
        th, tw = captured["sizes"][0]
        props = captured["proposals"][0].float().cpu() * torch.tensor(
            [img.shape[2] / tw, img.shape[1] / th, img.shape[2] / tw, img.shape[1] / th])
        dets = out["boxes"][out["scores"] >= score_thr].float().cpu()
        row = {"side": ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])).sqrt().numpy()}
        for n in top_n:
            row[f"rpn_recall@{n}"] = (box_iou(gt_boxes, props[:n]).max(1).values >= 0.5).numpy() if len(props) else 0
        row["rpn_best_iou"] = box_iou(gt_boxes, props).max(1).values.numpy() if len(props) else 0.0
        row["final_best_iou"] = box_iou(gt_boxes, dets).max(1).values.numpy() if len(dets) else 0.0
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
