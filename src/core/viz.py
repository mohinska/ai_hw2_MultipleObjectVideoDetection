"""Looking inside trained detectors: Grad-CAM and RoI embeddings of Faster R-CNN, deformable attention and
query embeddings of RT-DETR, and logging runs to Weights & Biases.

References: Grad-CAM (Selvaraju et al., arXiv 1610.02391); energy-based pointing game (Wang et al., Score-CAM,
arXiv 1910.01279); Deformable DETR (Zhu et al., arXiv 2010.04159); RT-DETR (Zhao et al., arXiv 2304.08069).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from . import CLASSES
from .cv import existing
from .data import KAGGLE_INPUT

FPN_LEVELS = {"0": "P2 (stride 4)", "1": "P3 (stride 8)", "2": "P4 (stride 16)", "3": "P5 (stride 32)"}
BACKBONE_STAGES = {"0": "C2 (stride 4)", "1": "C3 (stride 8)", "2": "C4 (stride 16)", "3": "C5 (stride 32)"}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- Faster R-CNN

def _frcnn_input(model, img, cfg):
    """Resized frame as the model input tensor, and the scale from original to model-input pixels."""
    from .tv import resize_long_side
    small, scale = resize_long_side(img, cfg["imgsz"])
    return torch.from_numpy(small).permute(2, 0, 1).float().div(255).to(_device()), scale


def frcnn_gradcam(model, img, box, cls, cfg):
    """Grad-CAM of one detection at the ResNet stages C2-C5 and the FPN levels P2-P5.

    The target is the score of the detection of class `cls` that overlaps `box` (xyxy, original pixels) most;
    for each map, the activations are weighted by their mean gradient and summed over channels (Grad-CAM).
    The RoI head pools each box from one FPN level chosen by its size, so the other FPN maps are zero; the
    backbone stages all receive gradient through the FPN.
    Returns ({map name: map in original pixels}, {"score", "iou"}) or (None, None) if nothing overlaps the box.
    """
    device = _device()
    model.to(device).eval()
    feats, sizes = {}, {}

    def keep(names):
        def hook(module, inputs, out):
            for k, v in out.items():
                if k in names:
                    v.retain_grad()
                    feats[names[k]] = v
        return hook

    def keep_sizes(module, inputs, out):
        sizes["padded"], sizes["image"] = out[0].tensors.shape[-2:], out[0].image_sizes[0]

    hooks = [model.backbone.body.register_forward_hook(keep(BACKBONE_STAGES)),
             model.backbone.register_forward_hook(keep(FPN_LEVELS)), model.transform.register_forward_hook(keep_sizes)]
    try:
        x, scale = _frcnn_input(model, img, cfg)
        with torch.enable_grad():
            out = model([x])[0]
            same = out["labels"] == cls + 1
            if not same.any():
                return None, None
            iou = box_iou(torch.as_tensor([box], dtype=torch.float32, device=device) * scale, out["boxes"].detach())[0]
            iou = torch.where(same, iou, torch.zeros_like(iou))
            idx = int(iou.argmax())
            if iou[idx] < 0.1:
                return None, None
            model.zero_grad()
            out["scores"][idx].backward()
    finally:
        for h in hooks:
            h.remove()

    H, W = img.shape[:2]
    (ph, pw), (ih, iw) = sizes["padded"], sizes["image"]
    cams = {}
    for name in [*BACKBONE_STAGES.values(), *FPN_LEVELS.values()]:
        if feats[name].grad is None:
            continue
        a, g = feats[name].detach()[0], feats[name].grad[0]
        cam = torch.relu((g.mean((1, 2))[:, None, None] * a).sum(0)).cpu().numpy()
        cam = cam[:int(np.ceil(cam.shape[0] * ih / ph)), :int(np.ceil(cam.shape[1] * iw / pw))]  # drop padding
        cams[name] = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    return cams, {"score": float(out["scores"][idx].detach()), "iou": float(iou[idx])}


@torch.no_grad()
def frcnn_roi_embeddings(model, img, boxes, cfg):
    """1024-d box-head features (RoIAlign on the FPN + two-layer head) of xyxy `boxes` in original pixels."""
    device = _device()
    model.to(device).eval()
    x, scale = _frcnn_input(model, img, cfg)
    images, _ = model.transform([x])
    features = model.backbone(images.tensors)
    ih, iw = images.image_sizes[0]
    r = torch.tensor([iw / x.shape[2], ih / x.shape[1]] * 2, device=device)
    b = torch.as_tensor(np.asarray(boxes, "float32"), device=device).reshape(-1, 4) * scale * r
    pooled = model.roi_heads.box_roi_pool(features, [b], images.image_sizes)
    return model.roi_heads.box_head(pooled).cpu().numpy()


def cam_concentration(cam, box, margin=16):
    """Share of the map's mass inside the box (grown by `margin` px) divided by the share of the frame the box
    covers: 1 = no more than a uniform map, larger = focused on the object (energy-based pointing game)."""
    H, W = cam.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(W, x2 + margin), min(H, y2 + margin)
    total = cam.sum()
    if total <= 0 or x2 <= x1 or y2 <= y1:
        return np.nan
    return (cam[y1:y2, x1:x2].sum() / total) / ((x2 - x1) * (y2 - y1) / (W * H))


def overlay(img, cam, alpha=0.5):
    """Heat map over the RGB frame."""
    m = cam / cam.max() if cam.max() > 0 else cam
    heat = cv2.cvtColor(cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


# ---------------------------------------------------------------- RT-DETR

def rtdetr_probe(net, img, imgsz=1280):
    """One RT-DETR forward pass with the decoder internals kept, all in original pixels.

    net: the nn.Module of an Ultralytics RT-DETR (`load_ultralytics(w).model`). The frame is stretched to
    imgsz x imgsz, as Ultralytics does for RT-DETR. Returns per query: boxes (xyxy), score, cls, emb (output of
    the last decoder layer) and, per decoder layer, the deformable-attention sampling points `xy`
    (queries x heads x points x 2), their weights `w` and the feature level of each point (`level`).
    """
    device = next(net.parameters()).device
    head = net.model[-1]
    attn = [layer.cross_attn for layer in head.decoder.layers]
    store = {"refer": [], "offsets": [], "weights": []}
    hooks = []
    for a in attn:
        hooks.append(a.register_forward_pre_hook(lambda m, args: store["refer"].append(args[1].detach())))
        hooks.append(a.sampling_offsets.register_forward_hook(lambda m, i, o: store["offsets"].append(o.detach())))
        hooks.append(a.attention_weights.register_forward_hook(lambda m, i, o: store["weights"].append(o.detach())))
    hooks.append(head.decoder.register_forward_hook(lambda m, i, o: store.__setitem__("out", o)))
    hooks.append(head.decoder.layers[head.decoder.eval_idx].register_forward_hook(
        lambda m, i, o: store.__setitem__("emb", o.detach())))
    H, W = img.shape[:2]
    x = torch.from_numpy(cv2.resize(img, (imgsz, imgsz))).permute(2, 0, 1).float().div(255)[None].to(device)
    try:
        with torch.no_grad():
            net.eval()(x)
    finally:
        for h in hooks:
            h.remove()

    scale = np.array([W, H], "float32")
    boxes, logits = store["out"][0][-1, 0].float().cpu().numpy(), store["out"][1][-1, 0].float().cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1) * np.tile(scale, 2)
    a0 = attn[0]
    n_pts = a0.n_levels * a0.n_points
    layers = []
    for refer, off, w in zip(store["refer"], store["offsets"], store["weights"]):
        nq = off.shape[1]
        off = off.view(1, nq, a0.n_heads, n_pts, 2)
        w = torch.softmax(w.view(1, nq, a0.n_heads, n_pts), -1)
        ref = refer[:, :, None, :, :]  # (1, nq, 1, 1, 4): centre x, y, width, height in [0, 1]
        loc = ref[..., :2] + off / a0.n_points * ref[..., 2:] * 0.5
        layers.append({"xy": loc[0].float().cpu().numpy() * scale, "w": w[0].float().cpu().numpy(),
                       "level": np.repeat(np.arange(a0.n_levels), a0.n_points)})
    return {"boxes": xyxy, "score": probs.max(1), "cls": probs.argmax(1), "emb": store["emb"][0].float().cpu().numpy(),
            "layers": layers}


def attention_stats(probe, queries):
    """Per decoder layer and query: weighted distance of the sampling points from the box centre relative to the
    box size, and the share of attention weight on each feature level."""
    rows = []
    for li, layer in enumerate(probe["layers"]):
        for q in queries:
            b = probe["boxes"][q]
            centre, size = (b[:2] + b[2:]) / 2, max(np.sqrt((b[2] - b[0]) * (b[3] - b[1])), 1.0)
            xy, w = layer["xy"][q].reshape(-1, 2), layer["w"][q].reshape(-1)
            dist = (np.linalg.norm(xy - centre, axis=1) * w).sum() / w.sum()
            level_w = layer["w"][q].sum(0)  # (points,) summed over heads
            share = {f"level {lv}": level_w[layer["level"] == lv].sum() / level_w.sum() for lv in np.unique(layer["level"])}
            rows.append({"layer": li + 1, "query": q, "size": size, "distance_rel": dist / size, **share})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Weights & Biases

def _curves(name, k, work):
    """Per-epoch curve of fold k: torchvision history csv or Ultralytics results.csv; columns epoch, mAP, losses."""
    f = existing(f"history_{name}_fold{k}.csv", work)
    if f is not None:
        return pd.read_csv(f)
    hits = [Path(work) / "runs" / f"{name}_fold{k}" / "results.csv"]
    hits += sorted(KAGGLE_INPUT.rglob(f"{name}_fold{k}/results.csv")) if KAGGLE_INPUT.exists() else []
    f = next((h for h in hits if h.exists()), None)
    if f is None:
        return None
    log = pd.read_csv(f).rename(columns=str.strip)
    return log.rename(columns={"metrics/mAP50-95(B)": "mAP", "metrics/mAP50(B)": "AP50"})


def _wandb_image(wandb, path, preds, gt_boxes, gt_cls, max_side=1280):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    s = min(1.0, max_side / max(img.shape[:2]))
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
    labels = dict(enumerate(CLASSES))

    def box(x1, y1, x2, y2, c, caption, score=None):
        d = {"position": {"minX": float(x1 * s), "minY": float(y1 * s), "maxX": float(x2 * s), "maxY": float(y2 * s)},
             "domain": "pixel",
             "class_id": int(c), "box_caption": caption}
        if score is not None:
            d["scores"] = {"score": float(score)}
        return d

    pred = [box(r.x, r.y, r.x + r.w, r.y + r.h, r.cls, f"{CLASSES[int(r.cls)]} {r.score:.2f}", r.score)
            for r in preds.itertuples()]
    truth = [box(*b, c, CLASSES[int(c)]) for b, c in zip(gt_boxes, gt_cls)]
    return wandb.Image(img, boxes={"predictions": {"box_data": pred, "class_labels": labels},
                                   "ground_truth": {"box_data": truth, "class_labels": labels}})


def wandb_log_run(name, work, frames, targets, project, n_folds, n_images=12, score_thr=0.3, seed=0):
    """One W&B run per experiment: config and environment, per-fold training curves, fold and out-of-fold metrics,
    and `n_images` held-out frames of fold 0 with ground truth and predictions (score >= score_thr)."""
    import wandb
    cv = json.loads(existing(f"cv_{name}.json", work).read_text())
    run = wandb.init(project=project, name=name, group=name.split("_")[0], config={**cv["cfg"], **cv.get("env", {})},
                     reinit=True)
    for k in range(n_folds):
        log = _curves(name, k, work)
        if log is None:
            continue
        run.define_metric(f"fold{k}/epoch")
        run.define_metric(f"fold{k}/*", step_metric=f"fold{k}/epoch")
        cols = [c for c in log.columns if c == "mAP" or c == "AP50" or "loss" in c]
        for row in log.itertuples(index=False):
            r = row._asdict()
            run.log({f"fold{k}/epoch": r["epoch"], **{f"fold{k}/{c.replace('/', '_')}": r[c] for c in cols}})
    run.summary.update({f"fold{k}/{m}": v for k, fm in cv["folds"].items() for m, v in fm.items()
                        if isinstance(v, (int, float))})
    if len(cv.get("folds_run", cv["folds"])) == n_folds:
        run.summary.update({f"oof/{m}": v for m, v in cv["oof"].items()})
    oof = existing(f"oof_{name}_fold0.parquet", work)
    if oof is not None:
        dets = pd.read_parquet(oof)
        held = frames[frames["fold"] == 0]
        held = held.sample(min(n_images, len(held)), random_state=seed)
        table = wandb.Table(columns=["frame", "image"])
        for r in held.itertuples():
            boxes, cls, _ = targets[r.uid]
            p = dets[(dets["key"] == r.key) & (dets["score"] >= score_thr)]
            table.add_data(r.key, _wandb_image(wandb, r.path, p, boxes, cls))
        run.log({"validation predictions (fold 0)": table})
    url = run.url
    run.finish()
    return url
