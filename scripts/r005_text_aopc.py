"""R005: text-side AOPC for occlusion attribution on the selection set.

Ranks tokens by leave-one-out deletion, then measures deletion/insertion AOPC
against a random-ranking baseline. Every sample is appended to a JSONL
checkpoint as soon as it finishes, so a killed session loses at most one sample.

Usage: python scripts/r005_text_aopc.py [--limit 50] [--restart]
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
from mmhs.faithfulness import content_positions, random_baseline, text_aopc  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
CHECKPOINT = os.path.join(OUT, "r005_text_occlusion.jsonl")
FINAL = os.path.join(OUT, "r005_text_occlusion.json")
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "r005_text_occlusion.log")


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_selection(limit: int | None) -> list:
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["selection"]
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)

    samples = []
    for tweet_id in ids:
        image_path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(image_path):
            continue
        entry = gt[str(tweet_id)]
        text = (entry.get("tweet_text") or "").strip()
        if not text:
            continue
        samples.append(Sample(
            tweet_id=str(tweet_id), text=text, ocr_text="",
            image_path=image_path, labels=normalize_labels(entry.get("labels")),
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
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if args.restart and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    done = read_checkpoint()
    samples = load_selection(args.limit or None)
    pending = [s for s in samples if s.tweet_id not in done]
    log(f"start: {len(samples)} selected, {len(done)} already done, {len(pending)} pending")

    if not pending:
        log("nothing to do")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, processor = load_model(os.environ["MMHS_MODEL_DIR"], device=device)
        wrapper = Wrapper(model, tokenizer, processor, device=device)
        log(f"model loaded on {device}")

        started = time.time()
        for index, sample in enumerate(pending):
            encoded = wrapper.encode_text([sample.text])
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            image = Image.open(sample.image_path).convert("RGB")
            pix = wrapper.encode_image([image])["pixel_values"]

            positions = content_positions(mask, tokenizer)
            if not positions:
                continue
            p0 = wrapper.target(ids, mask, pix).item()

            importances = []
            for position in positions:
                probe_mask = mask.clone()
                probe_mask[0, position] = 0
                importances.append(p0 - wrapper.target(ids, probe_mask, pix).item())
            ranking = [positions[i] for i in np.argsort(importances)[::-1]]

            ordered = text_aopc(wrapper, ids, mask, pix, ranking)
            random_scores = random_baseline(wrapper, ids, mask, pix, "text", positions,
                                             seed=int(sample.tweet_id) % 2**31)

            record = {
                "tweet_id": sample.tweet_id,
                "n_tokens": len(positions),
                "p0": p0,
                "label_strict": sample.binary_strict,
                "agreement": sample.agreement,
                "aopc_del": ordered["aopc_del"],
                "aopc_ins": ordered["aopc_ins"],
                "aopc_del_random": random_scores["aopc_del"],
                "aopc_ins_random": random_scores["aopc_ins"],
                "curve_del": ordered["curve_del"],
                "curve_ins": ordered["curve_ins"],
                "fractions": ordered["fractions"],
                "comp_20": ordered["comp_20"],
                "suff_20": ordered["suff_20"],
                "comp_50": ordered["comp_50"],
                "suff_50": ordered["suff_50"],
                "top_importance": float(max(importances)),
                "mean_importance": float(np.mean(importances)),
            }
            with open(CHECKPOINT, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            if (index + 1) % 25 == 0:
                rate = (index + 1) / max(time.time() - started, 1e-6)
                remaining = (len(pending) - index - 1) / rate
                log(f"{index + 1}/{len(pending)} done, {rate:.2f}/s, eta {remaining / 60:.1f} min")

    records = list(read_checkpoint().values())
    if not records:
        log("no records collected")
        return

    from scipy import stats

    def summarize(values):
        values = np.asarray(values)
        t_stat, p_value = stats.ttest_1samp(values, 0.0)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "ci95": [float(x) for x in stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values))],
            "t": float(t_stat),
            "p": float(p_value),
            "win_rate": float(np.mean(values > 0)),
        }

    summary = {
        "method": "occlusion (leave-one-out ranking)",
        "modality": "text",
        "n": len(records),
        "mean_aopc_del": float(np.mean([r["aopc_del"] for r in records])),
        "mean_aopc_del_random": float(np.mean([r["aopc_del_random"] for r in records])),
        "mean_aopc_ins": float(np.mean([r["aopc_ins"] for r in records])),
        "mean_aopc_ins_random": float(np.mean([r["aopc_ins_random"] for r in records])),
        "delta_del": summarize([r["aopc_del"] - r["aopc_del_random"] for r in records]),
        "delta_ins": summarize([r["aopc_ins"] - r["aopc_ins_random"] for r in records]),
    }
    with open(FINAL, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2, ensure_ascii=False)

    log(f"final: del {summary['mean_aopc_del']:.4f} vs random {summary['mean_aopc_del_random']:.4f} "
        f"(p={summary['delta_del']['p']:.2e}); ins {summary['mean_aopc_ins']:.4f} vs "
        f"{summary['mean_aopc_ins_random']:.4f} (p={summary['delta_ins']['p']:.2e}); n={len(records)}")


if __name__ == "__main__":
    main()
