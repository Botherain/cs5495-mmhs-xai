"""P1.3: faithfulness of the seven candidates on the 499-sample selection set.

For every method and sample: AOPC (mean drop over the deletion fractions), the
comprehensiveness / sufficiency pair at 20% and 50%, against a fixed random
ordering of the same units (3 permutations per sample). Text units are content
tokens, image units are the 7x7 cells; deletion replaces content (attention-mask
flip / mean colour), never truly removes it.

Outputs
  refine-logs/p1_faithfulness.jsonl  per sample, per method
  refine-logs/p1_matrix.json         aggregates + paired tests vs random

Usage: python scripts/p1_evaluate.py [--limit N]
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

from mmhs.attribution import GRID, IMAGE_METHODS, TEXT_METHODS, content_positions  # noqa: E402
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.faithfulness import (  # noqa: E402
    HEADLINE_FRACTIONS, image_aopc, random_ranking, text_aopc,
)
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p1_faithfulness.jsonl")
RANDOM_REPEATS = 3


def load_jsonl(path: str) -> dict:
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["id"]] = record
    return records


def load_samples(limit: int = 0) -> list:
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["selection"]
    samples = []
    for tweet_id in ids:
        path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(path):
            continue
        entry = gt[str(tweet_id)]
        samples.append(
            Sample(str(tweet_id), entry.get("tweet_text", ""), "", path, normalize_labels(entry["labels"]))
        )
    return samples[:limit] if limit else samples


def rank_from_values(positions: list, values: list) -> list:
    order = np.argsort(-np.asarray(values, dtype=np.float64))
    return [positions[index] for index in order]


def summarise(rows: list, methods: list, random_key: str) -> dict:
    from scipy.stats import wilcoxon

    summary = {}
    for method in methods:
        block = {}
        for metric in ("aopc_del", "aopc_ins", "comp_20", "comp_50", "suff_20", "suff_50"):
            values = np.array([row[method][metric] for row in rows], dtype=np.float64)
            reference = np.array([row[random_key][metric] for row in rows], dtype=np.float64)
            entry = {
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "std": float(np.nanstd(values)),
                "random_mean": float(np.nanmean(reference)),
                "diff_vs_random": float(np.nanmean(values - reference)),
                "n": int(np.isfinite(values).sum()),
            }
            paired = values - reference
            paired = paired[np.isfinite(paired)]
            if paired.size > 10 and np.any(paired != 0):
                test = wilcoxon(paired)
                entry["p_vs_random"] = float(test.pvalue)
            else:
                entry["p_vs_random"] = None
            block[metric] = entry
        summary[method] = block
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image

    text_records = load_jsonl(os.path.join(OUT, "p1_text_attr.jsonl"))
    image_records = load_jsonl(os.path.join(OUT, "p1_image_attr.jsonl"))
    print(f"[1] text attr={len(text_records)} image attr={len(image_records)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    samples = load_samples(args.limit)
    finished = {record["id"] for record in load_jsonl(JSONL).values()}
    todo = [
        sample for sample in samples
        if sample.tweet_id in text_records and sample.tweet_id in image_records
        and sample.tweet_id not in finished
    ]
    print(f"[2] samples={len(samples)} todo={len(todo)}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    started = time.time()
    with open(JSONL, "a", encoding="utf-8") as sink:
        for step, sample in enumerate(todo, start=1):
            encoded = wrapper.encode_text([sample.text])
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            pixel = wrapper.encode_image([Image.open(sample.image_path).convert("RGB")])["pixel_values"]
            positions = content_positions(mask)

            row = {"id": sample.tweet_id}
            text_record = text_records[sample.tweet_id]
            for name in TEXT_METHODS:
                ranking = rank_from_values(positions, text_record[name][:len(positions)])
                row[name] = text_aopc(wrapper, ids, mask, pixel, ranking)

            image_record = image_records[sample.tweet_id]
            for name in IMAGE_METHODS:
                ranking = list(np.argsort(-np.asarray(image_record[name], dtype=np.float64)))
                row[name] = image_aopc(wrapper, ids, mask, pixel, ranking)

            # fixed random orderings of the same units, averaged
            text_random, image_random = [], []
            for repeat in range(RANDOM_REPEATS):
                text_ranking = random_ranking(positions, repeats=1, seed=step * 17 + repeat)[0]
                text_random.append(text_aopc(wrapper, ids, mask, pixel, text_ranking))
                image_ranking = random_ranking(list(range(GRID * GRID)), repeats=1, seed=step * 31 + repeat)[0]
                image_random.append(image_aopc(wrapper, ids, mask, pixel, image_ranking))
            row["text_random"] = {
                key: float(np.mean([item[key] for item in text_random]))
                for key in ("aopc_del", "aopc_ins", "comp_20", "comp_50", "suff_20", "suff_50")
            }
            row["image_random"] = {
                key: float(np.mean([item[key] for item in image_random]))
                for key in ("aopc_del", "aopc_ins", "comp_20", "comp_50", "suff_20", "suff_50")
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()

            if step % 10 == 0 or step == len(todo):
                rate = step / max(time.time() - started, 1e-6)
                print(
                    f"    {step}/{len(todo)} {rate:.2f}/s eta {(len(todo) - step) / rate / 60:.1f} min",
                    flush=True,
                )

    rows = list(load_jsonl(JSONL).values())
    if not rows:
        print("[3] nothing to summarise", flush=True)
        return
    summary = {
        "n": len(rows),
        "fractions": list(HEADLINE_FRACTIONS),
        "text": summarise(rows, TEXT_METHODS, "text_random"),
        "image": summarise(rows, IMAGE_METHODS, "image_random"),
    }
    with open(os.path.join(OUT, "p1_matrix.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[3] n={len(rows)}")
    for modality, methods in (("text", TEXT_METHODS), ("image", IMAGE_METHODS)):
        print(f"    --- {modality} ---")
        for method in methods:
            block = summary[modality][method]["aopc_del"]
            comp = summary[modality][method]["comp_20"]
            suff = summary[modality][method]["suff_20"]
            print(
                f"    {method:11s} aopc={block['mean']:+.4f} (rand {block['random_mean']:+.4f}, "
                f"p={block['p_vs_random']}) comp20={comp['mean']:+.4f} suff20={suff['mean']:+.4f}"
            )


if __name__ == "__main__":
    main()
