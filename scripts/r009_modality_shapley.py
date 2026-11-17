"""R009/R010/R012: modality-level coalition attribution.

Four coalitions per sample (blank text / blank image / both), giving exact
two-player Shapley values, the interaction index, and the fusion-gate norms used
as an independent cross-check.

phi_T = 0.5 [v(T) - v(0)] + 0.5 [v(T,I) - v(I)]
phi_I = 0.5 [v(I) - v(0)] + 0.5 [v(T,I) - v(T)]
interaction = v(T,I) - v(T) - v(I) + v(0)

Usage: python scripts/r009_modality_shapley.py [--limit 100] [--restart]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
CHECKPOINT = os.path.join(OUT, "r009_modality_shapley.jsonl")
FINAL = os.path.join(OUT, "r009_modality_shapley.json")
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "r009_modality_shapley.log")


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_samples(limit: int | None) -> list:
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["main"]
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)

    samples = []
    for tweet_id in ids:
        image_path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(image_path):
            continue
        entry = gt[str(tweet_id)]
        samples.append(Sample(
            tweet_id=str(tweet_id), text=(entry.get("tweet_text") or "").strip(),
            ocr_text="", image_path=image_path,
            labels=normalize_labels(entry.get("labels")),
        ))
    return samples[:limit] if limit else samples


def read_checkpoint() -> dict:
    records = {}
    if not os.path.exists(CHECKPOINT):
        return records
    with open(CHECKPOINT, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[record["tweet_id"]] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    os.makedirs(OUT, exist_ok=True)
    if args.restart and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    done = read_checkpoint()
    samples = load_samples(args.limit or None)
    pending = [s for s in samples if s.tweet_id not in done]
    log(f"start: {len(samples)} selected, {len(done)} done, {len(pending)} pending")

    if pending:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, processor = load_model(os.environ["MMHS_MODEL_DIR"], device=device)
        wrapper = Wrapper(model, tokenizer, processor, device=device)
        blank_ids, blank_mask = wrapper.blank_text(1)
        log(f"model loaded on {device}")

        started = time.time()
        for index, sample in enumerate(pending):
            encoded = wrapper.encode_text([sample.text or ""])
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            image = Image.open(sample.image_path).convert("RGB")
            pix = wrapper.encode_image([image])["pixel_values"]
            zeros = wrapper.blank_image(pix)

            v0 = wrapper.target(blank_ids, blank_mask, zeros).item()
            v_t = wrapper.target(ids, mask, zeros).item()
            v_i = wrapper.target(blank_ids, blank_mask, pix).item()
            v_ti = wrapper.target(ids, mask, pix).item()

            phi_t = 0.5 * (v_t - v0) + 0.5 * (v_ti - v_i)
            phi_i = 0.5 * (v_i - v0) + 0.5 * (v_ti - v_t)
            interaction = v_ti - v_t - v_i + v0

            text_norm, image_norm = wrapper.gate_stats(ids, mask, pix)
            text_norm = float(text_norm[0])
            image_norm = float(image_norm[0])

            record = {
                "tweet_id": sample.tweet_id,
                "label_strict": sample.binary_strict,
                "agreement": sample.agreement,
                "v_empty": v0, "v_text": v_t, "v_image": v_i, "v_both": v_ti,
                "phi_text": phi_t, "phi_image": phi_i, "interaction": interaction,
                "completeness": phi_t + phi_i - (v_ti - v0),
                "gate_text_norm": text_norm, "gate_image_norm": image_norm,
            }
            with open(CHECKPOINT, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            if (index + 1) % 200 == 0:
                rate = (index + 1) / max(time.time() - started, 1e-6)
                log(f"{index + 1}/{len(pending)} done, {rate:.1f}/s, eta {(len(pending) - index - 1) / rate / 60:.1f} min")

    records = list(read_checkpoint().values())
    if not records:
        log("no records")
        return

    from scipy import stats

    phi_t = np.array([r["phi_text"] for r in records])
    phi_i = np.array([r["phi_image"] for r in records])
    inter = np.array([r["interaction"] for r in records])
    complete = np.array([r["completeness"] for r in records])
    ratio = np.array([r["gate_image_norm"] / max(r["gate_text_norm"] + r["gate_image_norm"], 1e-12) for r in records])
    phi_ratio = np.abs(phi_i) / np.maximum(np.abs(phi_t) + np.abs(phi_i), 1e-12)

    rho, p_rho = stats.spearmanr(phi_ratio, ratio)
    text_dominant = float(np.mean(np.abs(phi_t) > np.abs(phi_i)))
    image_null = float(np.mean(np.abs(phi_i) < 1e-3))
    image_negative = float(np.mean(phi_i < 0))
    interaction_positive = float(np.mean(inter > 0))

    summary = {
        "n": len(records),
        "mean_phi_text": float(phi_t.mean()),
        "mean_phi_image": float(phi_i.mean()),
        "median_phi_text": float(np.median(phi_t)),
        "median_phi_image": float(np.median(phi_i)),
        "mean_abs_phi_text": float(np.abs(phi_t).mean()),
        "mean_abs_phi_image": float(np.abs(phi_i).mean()),
        "text_dominant_rate": text_dominant,
        "image_negligible_rate": image_null,
        "image_negative_rate": image_negative,
        "interaction_positive_rate": interaction_positive,
        "max_completeness_error": float(np.abs(complete).max()),
        "spearman_gate_vs_phi_ratio": float(rho),
        "spearman_p": float(p_rho),
    }

    with open(FINAL, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2, ensure_ascii=False)

    log(f"n={summary['n']} | phi_text={summary['mean_phi_text']:.4f} phi_image={summary['mean_phi_image']:.4f}")
    log(f"text-dominant {text_dominant:.3f} | image<0 {image_negative:.3f} | interaction>0 {interaction_positive:.3f}")
    log(f"max completeness error {summary['max_completeness_error']:.2e}")
    log(f"gate vs phi ratio spearman {rho:.3f} (p={p_rho:.2e})")


if __name__ == "__main__":
    main()
