"""P0.2-P0.4: stratification stats, baseline inference, model gate.

Implements the pre-registered gate of proposal Section 7:

  1. five-head macro-F1 and macro ROC-AUC against the released 0.569 / 0.783;
  2. ROC within +-0.03 but F1 below 0.539 -> per-head threshold scan (0.2-0.8)
     decides between a threshold discrepancy and a label-reconstruction gap;
  3. ROC outside +-0.03 -> SigLIP2 becomes the audited model.

Outputs
  refine-logs/p0_stats.json              stratification / coverage statistics
  refine-logs/p0_baseline_probs.npz      3,000-sample main set probabilities
  refine-logs/p0_selection_probs.npz     499-sample selection set probabilities
  refine-logs/p0_gate.json               per-head metrics, scan, verdict

Usage: python scripts/p0_gate.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import USABLE_HEADS, Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
BATCH = 48

HEADS = ["racist", "sexist", "homophobe", "religion", "otherhate"]
PUBLISHED_THRESHOLDS = [0.3, 0.7, 0.5, 0.3, 0.55]
PUBLISHED_F1 = [0.6722, 0.5889, 0.7447, 0.2230, 0.6161]
PUBLISHED_AUC = [0.7646, 0.8097, 0.8817, 0.6175, 0.8424]
RELEASED = {"macro_f1": 0.568996, "macro_roc": 0.783188}
TOLERANCE = 0.03
F1_FLOOR = RELEASED["macro_f1"] - TOLERANCE  # 0.539


def load_split() -> tuple:
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        subset = json.load(handle)
    return gt, subset


def build(gt: dict, ids: list) -> list:
    samples = []
    for tweet_id in ids:
        tweet_id = str(tweet_id)
        path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(path):
            continue
        entry = gt[tweet_id]
        samples.append(
            Sample(
                tweet_id=tweet_id,
                text=entry.get("tweet_text", ""),
                ocr_text="",
                image_path=path,
                labels=normalize_labels(entry["labels"]),
            )
        )
    return samples


def agreement_score(sample: Sample) -> tuple:
    """Class-level agreement: 1.0 (3/3 same class), 2/3, 1/3 (all differ)."""
    labels = sample.labels
    votes = len(labels)
    unique = len(set(labels))
    if unique == 1:
        return 1.0, "all_agree"
    if unique == votes:
        return 1.0 / 3.0, "all_differ"
    return 2.0 / 3.0, "partial"


def ocr_available(tweet_id: str) -> bool:
    path = os.path.join(ROOT, "img_txt", f"{tweet_id}.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return bool(str(payload.get("img_text", "")).strip())


def describe_split(samples: list) -> dict:
    counts = {name: 0 for name in HEADS}
    buckets = {"all_agree": 0, "partial": 0, "all_differ": 0}
    scores = []
    votes = []
    for sample in samples:
        score, bucket = agreement_score(sample)
        scores.append(score)
        buckets[bucket] += 1
        votes.append(len(sample.labels))
        for label in set(sample.labels):
            if label > 0:
                counts[HEADS[label - 1]] += 1
    strict = np.mean([sample.binary_strict for sample in samples])
    any_hate = np.mean([int(any(label > 0 for label in sample.labels)) for sample in samples])
    ocr = np.mean([ocr_available(sample.tweet_id) for sample in samples])
    return {
        "n": len(samples),
        "hate_rate_majority": round(float(strict), 4),
        "hate_rate_any_annotator": round(float(any_hate), 4),
        "class_share_any": {name: round(counts[name] / max(len(samples), 1), 4) for name in HEADS},
        "agreement_buckets": buckets,
        "agreement_score_mean": round(float(np.mean(scores)), 4),
        "annotators_per_item": {
            "mean": round(float(np.mean(votes)), 3),
            "min": int(np.min(votes)),
            "max": int(np.max(votes)),
        },
        "ocr_text_available": round(float(ocr), 4),
    }


def infer(wrapper: Wrapper, samples: list, tag: str) -> np.ndarray:
    from PIL import Image

    probs = []
    started = time.time()
    for start in range(0, len(samples), BATCH):
        chunk = samples[start:start + BATCH]
        texts = [sample.text for sample in chunk]
        images = [Image.open(sample.image_path).convert("RGB") for sample in chunk]
        encoded_text = wrapper.encode_text(texts)
        encoded_image = wrapper.encode_image(images)
        with torch.no_grad():
            probs.append(
                wrapper.head_probs(
                    encoded_text["input_ids"],
                    encoded_text["attention_mask"],
                    encoded_image["pixel_values"],
                ).cpu().numpy()
            )
        done = start + len(chunk)
        if (start // BATCH) % 10 == 0:
            rate = done / max(time.time() - started, 1e-6)
            print(f"    [{tag}] {done}/{len(samples)} {time.time() - started:.0f}s {rate:.0f}/s", flush=True)
    return np.concatenate(probs, axis=0)


def pack(samples: list, probs: np.ndarray) -> dict:
    labels = np.zeros((len(samples), 5), dtype=int)
    for index, sample in enumerate(samples):
        for label in set(sample.labels):
            if label > 0:
                labels[index, label - 1] = 1
    return {
        "probs": probs,
        "labels": labels,
        "binary_strict": np.array([sample.binary_strict for sample in samples], dtype=int),
        "binary_any": (labels.sum(axis=1) > 0).astype(int),
        "agree_score": np.array([agreement_score(sample)[0] for sample in samples]),
        "ids": np.array([sample.tweet_id for sample in samples]),
        "target": probs[:, USABLE_HEADS].mean(axis=1),
    }


def head_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    per_head = {}
    for index, name in enumerate(HEADS):
        y = labels[:, index]
        if y.sum() == 0 or y.sum() == len(y):
            per_head[name] = {"auc": float("nan"), "f1": float("nan"), "prevalence": float(y.mean())}
            continue
        pred = (probs[:, index] > PUBLISHED_THRESHOLDS[index]).astype(int)
        per_head[name] = {
            "auc": float(roc_auc_score(y, probs[:, index])),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "prevalence": float(y.mean()),
            "published_auc": PUBLISHED_AUC[index],
            "published_f1": PUBLISHED_F1[index],
        }
    return per_head


def threshold_scan(probs: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import f1_score

    grid = np.round(np.arange(0.20, 0.801, 0.01), 2)
    scan = {}
    for index, name in enumerate(HEADS):
        y = labels[:, index]
        if y.sum() == 0 or y.sum() == len(y):
            scan[name] = {"best_threshold": None, "best_f1": float("nan")}
            continue
        best = (-1.0, None)
        for threshold in grid:
            score = f1_score(y, (probs[:, index] > threshold).astype(int), zero_division=0)
            if score > best[0]:
                best = (score, float(threshold))
        scan[name] = {
            "best_threshold": best[1],
            "best_f1": float(best[0]),
            "published_threshold": PUBLISHED_THRESHOLDS[index],
            "published_f1": PUBLISHED_F1[index],
        }
    return scan


def main() -> None:
    from sklearn.metrics import roc_auc_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[0] device={device}", flush=True)
    gt, subset = load_split()

    main_samples = build(gt, subset["main"])
    selection_samples = build(gt, subset["selection"])
    overlap = set(subset["main"]) & set(subset["selection"])
    print(f"[1] main={len(main_samples)} selection={len(selection_samples)} overlap={len(overlap)}", flush=True)

    stats = {
        "split": subset.get("split", "test"),
        "main": describe_split(main_samples),
        "selection": describe_split(selection_samples),
        "overlap_ids": len(overlap),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p0_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print("[2] main stats:", json.dumps(stats["main"], ensure_ascii=False), flush=True)

    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    main_probs = infer(wrapper, main_samples, "main")
    selection_probs = infer(wrapper, selection_samples, "selection")

    main_pack = pack(main_samples, main_probs)
    selection_pack = pack(selection_samples, selection_probs)
    np.savez(os.path.join(OUT, "p0_baseline_probs.npz"), **main_pack)
    np.savez(os.path.join(OUT, "p0_selection_probs.npz"), **selection_pack)
    print("[3] probabilities saved", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    per_head = head_metrics(main_probs, main_pack["labels"])
    scan = threshold_scan(main_probs, main_pack["labels"])

    macro_auc = float(np.nanmean([per_head[name]["auc"] for name in HEADS]))
    macro_f1 = float(np.nanmean([per_head[name]["f1"] for name in HEADS]))
    macro_f1_best = float(np.nanmean([scan[name]["best_f1"] for name in HEADS]))
    macro_auc_4 = float(np.nanmean([per_head[HEADS[i]]["auc"] for i in USABLE_HEADS]))
    macro_f1_4 = float(np.nanmean([per_head[HEADS[i]]["f1"] for i in USABLE_HEADS]))

    roc_ok = abs(macro_auc - RELEASED["macro_roc"]) <= TOLERANCE
    f1_ok = macro_f1 >= F1_FLOOR
    scan_ok = macro_f1_best >= F1_FLOOR

    if roc_ok and f1_ok:
        verdict = "REPRODUCED"
        reason = (
            f"macro ROC {macro_auc:.3f} within {TOLERANCE} of {RELEASED['macro_roc']:.3f} "
            f"and macro F1 {macro_f1:.3f} above the {F1_FLOOR:.3f} floor."
        )
    elif roc_ok and not f1_ok and scan_ok:
        verdict = "THRESHOLD_DISCREPANCY"
        reason = (
            f"macro F1 {macro_f1:.3f} below {F1_FLOOR:.3f} at the published thresholds, but the "
            f"per-head scan recovers {macro_f1_best:.3f}; the gap is a threshold-calibration issue."
        )
    elif roc_ok and not f1_ok and not scan_ok:
        verdict = "LABEL_RECONSTRUCTION_KEEP_MODEL"
        reason = (
            f"ROC within tolerance but F1 {macro_f1:.3f} (scan best {macro_f1_best:.3f}) stays below "
            f"{F1_FLOOR:.3f}: attributed to label reconstruction, checkpoint retained with an exemption."
        )
    else:
        verdict = "SWITCH_TO_SIGLIP2"
        reason = (
            f"macro ROC {macro_auc:.3f} outside {TOLERANCE} of {RELEASED['macro_roc']:.3f}: "
            f"SigLIP2 becomes the audited model and the replication role is dropped."
        )

    gate = {
        "n": len(main_samples),
        "per_head": per_head,
        "macro_auc_5": macro_auc,
        "macro_f1_5": macro_f1,
        "macro_auc_4_usable": macro_auc_4,
        "macro_f1_4_usable": macro_f1_4,
        "published": RELEASED,
        "published_per_head": {"f1": PUBLISHED_F1, "auc": PUBLISHED_AUC, "thresholds": PUBLISHED_THRESHOLDS},
        "tolerance": TOLERANCE,
        "f1_floor": F1_FLOOR,
        "threshold_scan": scan,
        "macro_f1_scan_best": macro_f1_best,
        "roc_within_tolerance": bool(roc_ok),
        "f1_above_floor": bool(f1_ok),
        "scan_recovers_floor": bool(scan_ok),
        "verdict": verdict,
        "reason": reason,
        "target_auc_strict": float(roc_auc_score(main_pack["binary_strict"], main_pack["target"])),
    }
    with open(os.path.join(OUT, "p0_gate.json"), "w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, ensure_ascii=False)

    print(f"[4] macro ROC-AUC (5 heads) = {macro_auc:.4f} (released {RELEASED['macro_roc']:.4f})")
    print(f"    macro F1      (5 heads) = {macro_f1:.4f} (released {RELEASED['macro_f1']:.4f}, floor {F1_FLOOR:.3f})")
    print(f"    macro F1 scan-best      = {macro_f1_best:.4f}")
    print(f"    4 usable heads: AUC {macro_auc_4:.4f}  F1 {macro_f1_4:.4f}")
    for name in HEADS:
        row = per_head[name]
        print(
            f"    {name:11s} auc={row['auc']:.3f} (pub {row['published_auc']:.3f}) "
            f"f1={row['f1']:.3f} (pub {row['published_f1']:.3f}) prev={row['prevalence']:.3f} "
            f"scan={scan[name]['best_f1']:.3f}@{scan[name]['best_threshold']}"
        )
    print(f"[5] verdict = {verdict}")
    print(f"    {reason}")


if __name__ == "__main__":
    main()
