"""Attribution toolkit for the audited checkpoint (P1).

Everything in P1 runs on frozen tower features where possible: the image tower is
encoded once per sample and only the perturbed text tower is re-run for the three
text methods, and vice versa for the four image methods. That keeps 500 samples x
thousands of perturbations inside a few minutes.

Unit definitions, fixed before any measurement:

* text  -- one unit per *content* BPE token (the two CLIP sentinels excluded).
           Deletion flips attention_mask, never input_ids: CLIP pools the first
           EOS by argmax, so writing a pad id into the sentence would move the
           pooled slot and change the target for the wrong reason.
* image -- one unit per cell of the model's own 7x7 patch grid; deletion sets the
           cell to the mean colour (zeros in CLIP-normalised space).

Methods
  text : occlusion (leave-one-out), LIME (sampled masks + weighted ridge),
         KernelSHAP (sampled coalitions + SHAP kernel).
  image: occlusion, Grad-CAM (last vision block), IG (32 steps from mean colour),
         Score-CAM (7x7 upsampled masks).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

USABLE_HEADS = (0, 1, 2, 4)
TEXT_METHODS = ("occlusion", "lime", "kernelshap")
IMAGE_METHODS = ("occlusion", "gradcam", "ig", "scorecam")
MAX_LEN = 77
GRID = 7


# fusion path without the towers
def fusion_probs(model, tfeat_raw: torch.Tensor, vfeat_raw: torch.Tensor,
                 text_present: Optional[torch.Tensor] = None,
                 image_present: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Replicates MultiTaskClassifier.forward from the two pooler outputs."""
    if tfeat_raw.dim() == 1:
        tfeat_raw = tfeat_raw.unsqueeze(0)
    if vfeat_raw.dim() == 1:
        vfeat_raw = vfeat_raw.unsqueeze(0)
    tfeat = model.proj_t(tfeat_raw)
    vfeat = model.proj_i(vfeat_raw)
    batch = tfeat.shape[0]
    device = tfeat.device
    if text_present is None:
        text_present = torch.ones(batch, device=device)
    if image_present is None:
        image_present = torch.ones(batch, device=device)
    presence = torch.stack([text_present, image_present], dim=1)
    zt = torch.tanh(model.g_t(tfeat))
    zi = torch.tanh(model.g_i(vfeat))
    gate = torch.sigmoid(model.gate(torch.cat([tfeat, vfeat, presence], dim=1)))
    fused = torch.where(
        (image_present < 0.5).unsqueeze(1), zt,
        torch.where((text_present < 0.5).unsqueeze(1), zi, gate * zt + (1.0 - gate) * zi),
    )
    shared = model.shared_head(fused)
    logits = torch.stack([head(shared).squeeze(-1) for head in model.heads], dim=1)
    return torch.sigmoid(logits)


def target_of(probs: torch.Tensor) -> torch.Tensor:
    return probs[..., list(USABLE_HEADS)].mean(dim=-1)


@torch.no_grad()
def encode_features(wrapper, texts: Sequence[str], images: Sequence) -> tuple:
    """Pooler features for both towers, batched."""
    model = wrapper.model
    texts = list(texts)
    images = list(images)
    text_feats, image_feats = [], []
    for start in range(0, len(texts), 64):
        encoded = wrapper.encode_text(texts[start:start + 64])
        text_feats.append(
            model._encode_text(encoded["input_ids"].to(wrapper.device),
                               encoded["attention_mask"].to(wrapper.device)).cpu()
        )
    for start in range(0, len(images), 64):
        encoded = wrapper.encode_image(images[start:start + 64])
        image_feats.append(model._encode_image(encoded["pixel_values"].to(wrapper.device)).cpu())
    return torch.cat(text_feats), torch.cat(image_feats)


# text: attribution over content tokens
def content_positions(mask_row: torch.Tensor) -> List[int]:
    """Positions of real tokens, excluding the two CLIP sentinels.

    Accepts either a 77-slot row or a (1, 77) batch row; the batch form is what
    the tokenizer returns, and silently returning an empty list for it is exactly
    the kind of bug that makes every method look identical.
    """
    if mask_row.dim() == 2:
        mask_row = mask_row[0]
    flags = mask_row.tolist()
    positions = [i for i, flag in enumerate(flags) if flag == 1]
    if len(positions) <= 2:
        return positions
    return positions[1:-1]


@torch.no_grad()
def text_batch_probs(wrapper, ids_row: torch.Tensor, mask_row: torch.Tensor,
                     positions: Sequence[int], presence: torch.Tensor,
                     vfeat: torch.Tensor, batch: int = 256) -> torch.Tensor:
    """Target probability for each row of a binary token-presence matrix."""
    model = wrapper.model
    device = wrapper.device
    presence = presence.to(device).float()
    vfeat = vfeat.to(device)
    if vfeat.dim() == 1:
        vfeat = vfeat.unsqueeze(0)
    out = []
    for start in range(0, presence.shape[0], batch):
        chunk = presence[start:start + batch]
        size = chunk.shape[0]
        ids = ids_row.to(device).repeat(size, 1)
        mask = mask_row.to(device).repeat(size, 1).clone()
        mask[:, list(positions)] = chunk.long()
        tfeat = model._encode_text(ids, mask)
        probs = fusion_probs(model, tfeat, vfeat.expand(size, -1))
        out.append(target_of(probs).cpu())
    return torch.cat(out)


def text_occlusion(wrapper, ids_row, mask_row, positions, vfeat) -> np.ndarray:
    """Leave-one-out: v(full) - v(without token j)."""
    count = len(positions)
    if count == 0:
        return np.zeros(0)
    presence = torch.ones(1, count)
    full = text_batch_probs(wrapper, ids_row, mask_row, positions, presence, vfeat).item()
    presence = torch.ones(count, count)
    presence[torch.arange(count), torch.arange(count)] = 0.0
    without = text_batch_probs(wrapper, ids_row, mask_row, positions, presence, vfeat)
    return (full - without).numpy()


def _wls(design: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least squares with intercept, the shared LIME / KernelSHAP solve."""
    design = np.asarray(design, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    design = np.hstack([np.ones((design.shape[0], 1)), design])
    weighted = design * np.sqrt(weights)[:, None]
    lhs = weighted.T @ weighted
    rhs = weighted.T @ (values * np.sqrt(weights))
    lhs += np.eye(lhs.shape[0]) * 1e-6
    solution = np.linalg.solve(lhs, rhs)
    return solution[1:]


def text_lime(wrapper, ids_row, mask_row, positions, vfeat,
              num_samples: int = 1500, seed: int = 0, kernel_width: float = 25.0) -> np.ndarray:
    """Local surrogate over token-presence masks.

    LIME's own text defaults are used: Euclidean distance on the binary
    representation and the sqrt-exponential kernel with kernel_width 25, i.e.
    w = exp(-d^2 / (2 * 25^2)). A narrower kernel (0.25 * sqrt(n), the tabular
    default) collapses onto the near-full coalitions and returns almost uniform
    positive coefficients on this model.
    """
    count = len(positions)
    if count == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    presence = rng.integers(0, 2, size=(num_samples, count)).astype(np.float32)
    presence[0] = 1.0
    presence = np.unique(presence, axis=0)
    values = text_batch_probs(
        wrapper, ids_row, mask_row, positions, torch.from_numpy(presence), vfeat
    ).numpy()
    distance = np.sqrt(((1.0 - presence) ** 2).sum(axis=1))
    weights = np.exp(-(distance ** 2) / (2.0 * kernel_width ** 2))
    return _wls(presence, values, weights)


def text_kernelshap(wrapper, ids_row, mask_row, positions, vfeat,
                    num_samples: int = 2048, seed: int = 0) -> np.ndarray:
    """KernelSHAP: sampled coalitions, SHAP kernel, WLS for the Shapley values.

    Coalitions come in complementary pairs so that v(S) and v(N\\S) are both
    observed; this halves the variance of the paired estimator. Weights follow
    the SHAP kernel (M-1)/(C(M,|S|)|S|(M-|S|)) written in log space.
    """
    count = len(positions)
    if count == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    half = max(1, num_samples // 2)
    sizes = rng.integers(1, count, size=half)
    masks = np.zeros((half, count), dtype=np.float32)
    for index, size in enumerate(sizes):
        chosen = rng.choice(count, size=size, replace=False)
        masks[index, chosen] = 1.0
    presence = np.vstack([masks, 1.0 - masks, np.ones((1, count)), np.zeros((1, count))]).astype(np.float32)
    presence = np.unique(presence, axis=0)
    values = text_batch_probs(
        wrapper, ids_row, mask_row, positions, torch.from_numpy(presence), vfeat
    ).numpy()

    size = presence.sum(axis=1)
    from math import lgamma
    log_binom = np.array([lgamma(count + 1) - lgamma(s + 1) - lgamma(count - s + 1) for s in size])
    with np.errstate(divide="ignore"):
        log_weight = np.log(count - 1) - log_binom - np.log(np.clip(size, 1, None)) - np.log(np.clip(count - size, 1, None))
    weight = np.exp(log_weight)
    # v(empty) and v(full) carry infinite SHAP weight: they enforce the efficiency
    # constraint, so keep them at a large finite value instead of dropping them.
    finite = weight[np.isfinite(weight)]
    ceiling = 1e3 * (finite.max() if finite.size else 1.0)
    weight[~np.isfinite(weight)] = ceiling
    return _wls(presence, values, weight)


# image: attribution over the 7x7 patch grid
def patch_slices(image_size: int = 224, grid: int = GRID) -> List[tuple]:
    step = image_size // grid
    return [
        (row * step, (row + 1) * step, col * step, (col + 1) * step)
        for row in range(grid)
        for col in range(grid)
    ]


def _apply_patch_mask(pixel: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Zero (mean colour) every cell whose keep-mask is 0. keep: (grid*grid,)."""
    out = pixel.clone()
    for index, (r0, r1, c0, c1) in enumerate(patch_slices(pixel.shape[-1], GRID)):
        if keep[index] == 0:
            out[..., r0:r1, c0:c1] = 0.0
    return out


@torch.no_grad()
def image_batch_probs(wrapper, pixel: torch.Tensor, tfeat: torch.Tensor,
                      keep: torch.Tensor, batch: int = 128) -> torch.Tensor:
    """Target probability for each row of a binary patch keep-mask."""
    model = wrapper.model
    device = wrapper.device
    keep = keep.to(device).float()
    tfeat = tfeat.to(device)
    if tfeat.dim() == 1:
        tfeat = tfeat.unsqueeze(0)
    pixel = pixel.to(device)
    out = []
    for start in range(0, keep.shape[0], batch):
        chunk = keep[start:start + batch]
        stacked = torch.stack([_apply_patch_mask(pixel, row) for row in chunk])
        vfeat = model._encode_image(stacked)
        probs = fusion_probs(model, tfeat.expand(chunk.shape[0], -1), vfeat)
        out.append(target_of(probs).cpu())
    return torch.cat(out)


def _as_batch_pair(pixel, tfeat):
    if pixel.dim() == 3:
        pixel = pixel.unsqueeze(0)
    if tfeat.dim() == 1:
        tfeat = tfeat.unsqueeze(0)
    return pixel, tfeat


@torch.no_grad()
def _probs_pairs(wrapper, pixels, tfeats, keeps, batch: int = 128) -> torch.Tensor:
    """Target probability for paired (image, patch-keep) rows."""
    model = wrapper.model
    device = wrapper.device
    pixels = pixels.to(device)
    tfeats = tfeats.to(device)
    keeps = keeps.to(device)
    out = []
    for start in range(0, pixels.shape[0], batch):
        block = range(start, min(start + batch, pixels.shape[0]))
        stacked = torch.stack([_apply_patch_mask(pixels[i], keeps[i]) for i in block])
        vfeat = model._encode_image(stacked)
        probs = fusion_probs(model, tfeats[start:start + batch], vfeat)
        out.append(target_of(probs).cpu())
    return torch.cat(out)


def image_occlusion(wrapper, pixel, tfeat) -> np.ndarray:
    """Leave-one-patch-out: v(full) - v(without patch j), batched over samples."""
    pixel, tfeat = _as_batch_pair(pixel, tfeat)
    size = pixel.shape[0]
    count = GRID * GRID
    full = _probs_pairs(wrapper, pixel, tfeat, torch.ones(size, count)).repeat_interleave(count)
    keep = torch.ones(count, count)
    keep[torch.arange(count), torch.arange(count)] = 0.0
    keep_all = keep.unsqueeze(0).expand(size, count, count).reshape(-1, count)
    pixels = pixel.unsqueeze(1).expand(size, count, *pixel.shape[1:]).reshape(-1, *pixel.shape[1:])
    tfeats = tfeat.unsqueeze(1).expand(size, count, tfeat.shape[-1]).reshape(-1, tfeat.shape[-1])
    without = _probs_pairs(wrapper, pixels, tfeats, keep_all)
    return (full - without).reshape(size, count).numpy()


def _patch_reduce(cam: torch.Tensor) -> torch.Tensor:
    """Average-pool a 224x224 map into 7x7 cells; pass 7x7 through unchanged."""
    if cam.shape[-1] != GRID:
        cam = F.adaptive_avg_pool2d(cam.unsqueeze(1), GRID).squeeze(1)
    return cam.flatten(1)


def image_gradcam(wrapper, pixel, tfeat, layer: int = -1) -> np.ndarray:
    """Grad-CAM on the last vision block; the backward of the summed target gives
    each sample its own gradient because samples are independent."""
    pixel, tfeat = _as_batch_pair(pixel, tfeat)
    model = wrapper.model
    device = wrapper.device
    blocks = model.tower_img.encoder.layers
    store: Dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        activation = output[0] if isinstance(output, tuple) else output
        activation.retain_grad()
        store["act"] = activation

    handle = blocks[layer].register_forward_hook(hook)
    try:
        vfeat = model._encode_image(pixel.to(device))
        score = target_of(fusion_probs(model, tfeat.to(device), vfeat)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        activation = store["act"]
        gradient = activation.grad
    finally:
        handle.remove()

    weights = gradient.mean(dim=1, keepdim=True)
    cam = F.relu((weights * activation).sum(dim=-1))[:, 1:]
    return cam.detach().cpu().reshape(pixel.shape[0], GRID, GRID).flatten(1).numpy()


def image_ig(wrapper, pixel, tfeat, steps: int = 32) -> np.ndarray:
    """Integrated gradients from the mean colour; pixel attribution is summed
    over channels, taken absolute, then averaged into the 7x7 cells."""
    pixel, tfeat = _as_batch_pair(pixel, tfeat)
    model = wrapper.model
    device = wrapper.device
    pixel = pixel.to(device)
    tfeat = tfeat.to(device)
    baseline = torch.zeros_like(pixel)
    total = torch.zeros_like(pixel)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]:
        interpolated = (baseline + alpha * (pixel - baseline)).clone().requires_grad_(True)
        vfeat = model._encode_image(interpolated)
        score = target_of(fusion_probs(model, tfeat, vfeat)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        total += interpolated.grad.detach()
    attribution = ((pixel - baseline) * total / steps).detach().sum(dim=1).abs()
    return _patch_reduce(attribution).cpu().numpy()


def image_scorecam(wrapper, pixel, tfeat, batch: int = 196) -> np.ndarray:
    """Score-CAM with 7x7 upsampled masks: the mask itself is the feature map and
    its revealed-region probability is the weight."""
    pixel, tfeat = _as_batch_pair(pixel, tfeat)
    model = wrapper.model
    device = wrapper.device
    size = pixel.shape[0]
    count = GRID * GRID
    step = pixel.shape[-1] // GRID
    base = torch.zeros(count, 1, pixel.shape[-2], pixel.shape[-1])
    for index in range(count):
        row, col = divmod(index, GRID)
        base[index, :, row * step:(row + 1) * step, col * step:(col + 1) * step] = 1.0
    masks = F.interpolate(base, size=pixel.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    masks = masks.to(device)

    pixel = pixel.to(device)
    tfeat = tfeat.to(device)
    scores = []
    with torch.no_grad():
        for start in range(0, count, batch):
            chunk = masks[start:start + batch]
            masked = (pixel.unsqueeze(1) * chunk.unsqueeze(0)).reshape(-1, *pixel.shape[1:])
            vfeat = model._encode_image(masked)
            tfeats = tfeat.unsqueeze(1).expand(size, chunk.shape[0], tfeat.shape[-1]).reshape(-1, tfeat.shape[-1])
            probs = fusion_probs(model, tfeats, vfeat)
            scores.append(target_of(probs).reshape(size, chunk.shape[0]).cpu())
    weights = torch.cat(scores, dim=1)
    weights = weights - weights.min(dim=1, keepdim=True).values
    weights = weights / weights.max(dim=1, keepdim=True).values.clamp_min(1e-9)
    flat = masks.flatten(1).cpu()
    cam = torch.einsum("bm,mk->bk", weights, flat)
    return _patch_reduce(cam).numpy()


# evaluation helpers
def descending(values: Sequence[float]) -> List[int]:
    return [int(index) for index in np.argsort(-np.asarray(values, dtype=np.float64))]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def topk_overlap(a: Sequence[float], b: Sequence[float], fraction: float = 0.1) -> float:
    count = max(1, int(round(len(a) * fraction)))
    set_a = set(descending(a)[:count])
    set_b = set(descending(b)[:count])
    return len(set_a & set_b) / count
