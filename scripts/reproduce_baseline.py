"""R003: reproduce the released checkpoint's reported metrics on our subset.

Reports per-head ROC-AUC and macro-F1 with the thresholds shipped in
inference_config.json, plus the target-probability AUROC against the strict
binary label. Saves predictions for later blocks.

Usage: python scripts/reproduce_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.data import Sample, describe, normalize_labels  # noqa: E402
from mmhs.model import USABLE_HEADS, Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
THRESHOLDS = [0.3, 0.7, 0.5, 0.3, 0.55]
TASK_NAMES = ["racist", "sexist", "homophobe", "religion", "otherhate"]
BATCH = 32


def collect() -> list:
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["main"]

    samples = []
    for tweet_id in ids:
        image_path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(image_path):
            continue
        entry = gt[str(tweet_id)]
        samples.append(
            Sample(
                tweet_id=str(tweet_id),
                text=entry.get("tweet_text", ""),
                ocr_text="",
                image_path=image_path,
                labels=normalize_labels(entry["labels"]),
            )
        )
    return samples


def main() -> None:
    from PIL import Image
    from sklearn.metrics import f1_score, roc_auc_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    samples = collect()
    stats = describe(samples)
    print("[1] subset:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items() if k != "class_share"})

    probs = []
    started = time.time()
    for start in range(0, len(samples), BATCH):
        chunk = samples[start:start + BATCH]
        texts = [s.text for s in chunk]
        images = [Image.open(s.image_path).convert("RGB") for s in chunk]
        encoded_text = wrapper.encode_text(texts)
        encoded_image = wrapper.encode_image(images)
        probs.append(wrapper.head_probs(
            encoded_text["input_ids"], encoded_text["attention_mask"], encoded_image["pixel_values"]
        ).numpy())
        if (start // BATCH) % 20 == 0:
            print(f"    {start + len(chunk)}/{len(samples)} {time.time() - started:.0f}s", flush=True)

    probs = np.concatenate(probs, axis=0)
    max_votes = max(len(s.labels) for s in samples)
    labels = np.array([tuple(list(s.labels) + [0] * (max_votes - len(s.labels))) for s in samples])
    binary_any = (labels > 0).any(axis=1).astype(int)
    binary_strict = np.array([s.binary_strict for s in samples])

    print(f"[2] inference {len(samples)} samples in {time.time() - started:.0f}s")

    per_head = {}
    for index, name in enumerate(TASK_NAMES):
        y = (labels == index + 1).any(axis=1).astype(int)  # 0 = NotHate, 1..5 = classes
        if y.sum() == 0 or y.sum() == len(y):
            per_head[name] = {"auc": float("nan"), "f1": float("nan"), "prevalence": float(y.mean())}
            continue
        pred = (probs[:, index] > THRESHOLDS[index]).astype(int)
        per_head[name] = {
            "auc": float(roc_auc_score(y, probs[:, index])),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "prevalence": float(y.mean()),
        }

    macro_auc = float(np.nanmean([v["auc"] for v in per_head.values()]))
    macro_f1 = float(np.nanmean([v["f1"] for v in per_head.values()]))

    target = probs[:, USABLE_HEADS].mean(axis=1)
    target_auc_strict = float(roc_auc_score(binary_strict, target))
    target_auc_any = float(roc_auc_score(binary_any, target))

    report = {
        "n": len(samples),
        "dataset_stats": stats,
        "per_head": per_head,
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "target_auc_strict": target_auc_strict,
        "target_auc_any": target_auc_any,
        "published": {"macro_f1": 0.569, "macro_roc": 0.783},
        "caveat": "labels rebuilt as any-annotator class membership; published numbers use the authors' own pipeline",
    }

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "baseline_probs.npz"),
             probs=probs, labels=labels, binary_strict=binary_strict,
             ids=np.array([s.tweet_id for s in samples]))
    with open(os.path.join(OUT, "baseline_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"[3] macro ROC-AUC = {macro_auc:.3f} (published 0.783)")
    print(f"    macro F1      = {macro_f1:.3f} (published 0.569)")
    print(f"    target AUROC  = {target_auc_strict:.3f} (strict) / {target_auc_any:.3f} (any)")
    for name, values in per_head.items():
        print(f"    {name:11s} auc={values['auc']:.3f} f1={values['f1']:.3f} prevalence={values['prevalence']:.3f}")


if __name__ == "__main__":
    main()
