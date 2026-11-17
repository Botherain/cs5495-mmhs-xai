"""P4: failure modes --- does ablating the units an explanation blames flip the decision?

Attributions are the P1 winner's, recomputed on the full analysis set (P1.7):
occlusion, content tokens for text, 7x7 cells for image. The top 10% of units is
removed (attention-mask flip / mean colour) and the decision set is recomputed
from the checkpoint's own published thresholds over the four usable heads.

Three ablations (text only, image only, both) are compared with random orderings
of the same number of units (3 repeats each). Every sample is also assigned to one
of five outcome classes against the strict-majority label set, so the flip rates
can be reported per failure mode rather than as one pooled number.

Outputs
  refine-logs/p4_flip.jsonl   per sample
  refine-logs/p4_flip.json    aggregates, McNemar tests, per-class rates

Usage: python scripts/p4_flip.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import GRID, USABLE_HEADS, content_positions  # noqa: E402
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p4_flip.jsonl")
RANDOM_REPEATS = 3
K_FRACTION = 0.1
THRESHOLDS = np.array([0.3, 0.7, 0.5, 0.3, 0.55])


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
        ids = json.load(handle)["main"]
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


def decision_set(probs) -> frozenset:
    return frozenset(index for index in USABLE_HEADS if probs[index] > THRESHOLDS[index])


def truth_set(sample: Sample) -> frozenset:
    """Strict majority per head: more than half of the annotators chose the class."""
    votes = sample.labels
    counts = {}
    for label in votes:
        counts[label] = counts.get(label, 0) + 1
    return frozenset(index for index, label in enumerate([1, 2, 3, 4, 5])
                     if counts.get(label, 0) * 2 > len(votes))


def outcome_class(truth: frozenset, predicted: frozenset) -> str:
    if truth == predicted:
        return "correct"
    if not truth:
        return "false_positive"
    if not predicted:
        return "false_negative"
    if truth < predicted:
        return "over_trigger"
    return "misclass"


def ablate_text(mask, chosen) -> torch.Tensor:
    probe = mask.clone()
    for position in chosen:
        probe[0, position] = 0
    return probe


def ablate_image(pix, chosen) -> torch.Tensor:
    probe = pix.clone()
    step = pix.shape[-1] // GRID
    for index in chosen:
        row, col = divmod(int(index), GRID)
        probe[0, :, row * step:(row + 1) * step, col * step:(col + 1) * step] = 0.0
    return probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image

    text_records = load_jsonl(os.path.join(OUT, "p1d_text_attr_test.jsonl"))
    image_records = load_jsonl(os.path.join(OUT, "p1d_image_attr_test.jsonl"))
    print(f"[1] text={len(text_records)} image={len(image_records)}", flush=True)

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
    if not todo:
        summarise()
        return

    rng = np.random.default_rng(0)
    with open(JSONL, "a", encoding="utf-8") as sink:
        for step, sample in enumerate(todo, start=1):
            encoded = wrapper.encode_text([sample.text])
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            pix = wrapper.encode_image([Image.open(sample.image_path).convert("RGB")])["pixel_values"]
            base = wrapper.head_probs(ids, mask, pix)[0].tolist()
            positions = content_positions(mask)
            base_set = decision_set(base)
            truth = truth_set(sample)

            row = {
                "id": sample.tweet_id,
                "base": base,
                "base_set": sorted(base_set),
                "truth_set": sorted(truth),
                "outcome": outcome_class(truth, base_set),
                "k_fraction": K_FRACTION,
            }

            text_record = text_records[sample.tweet_id]
            positions = [int(position) for position in text_record.get("positions") or positions]
            text_values = np.asarray(text_record.get("values", text_record.get("occlusion")), dtype=float)
            k_text = int(max(1, round(len(positions) * K_FRACTION)))
            top_text = [positions[index] for index in np.argsort(-text_values)[:k_text]] if len(positions) else []

            image_record = image_records[sample.tweet_id]
            image_values = np.asarray(image_record.get("values", image_record.get("occlusion")), dtype=float)
            k_image = int(max(1, round(len(image_values) * K_FRACTION)))
            top_image = [int(index) for index in np.argsort(-image_values)[:k_image]]

            def evaluate(choice_text, choice_image) -> dict:
                probe_mask = ablate_text(mask, choice_text) if choice_text else mask
                probe_pix = ablate_image(pix, choice_image) if choice_image else pix
                probs = wrapper.head_probs(ids, probe_mask, probe_pix)[0].tolist()
                return {"probs": probs, "flip": decision_set(probs) != base_set}

            modes = {
                "text": (top_text, []),
                "image": ([], top_image),
                "both": (top_text, top_image),
            }
            for name, (choice_text, choice_image) in modes.items():
                result = evaluate(choice_text, choice_image)
                row[f"{name}_flip"] = bool(result["flip"])
                row[f"{name}_probs"] = result["probs"]

            for name, size, sampler in (
                ("text", k_text, lambda: [positions[index] for index in
                                          rng.choice(len(positions), size=k_text, replace=False)]
                 if len(positions) else []),
                ("image", k_image, lambda: rng.choice(GRID * GRID, size=k_image, replace=False).tolist()),
                ("both", None, None),
            ):
                flips = []
                for _ in range(RANDOM_REPEATS):
                    if name == "both":
                        choice_text = ([positions[index] for index in
                                        rng.choice(len(positions), size=k_text, replace=False)]
                                       if len(positions) else [])
                        choice_image = rng.choice(GRID * GRID, size=k_image, replace=False).tolist()
                    elif name == "text":
                        choice_text, choice_image = sampler(), []
                    else:
                        choice_text, choice_image = [], sampler()
                    flips.append(evaluate(choice_text, choice_image)["flip"])
                row[f"{name}_random_flip"] = float(np.mean(flips))

            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if step % 100 == 0 or step == len(todo):
                print(f"    {step}/{len(todo)}", flush=True)

    summarise()


def summarise() -> None:
    from scipy.stats import binomtest

    rows = list(load_jsonl(JSONL).values())
    if not rows:
        return

    def exact_mcnemar(top, random) -> float:
        """Exact McNemar on the discordant pairs (binomial test, p=0.5)."""
        top = np.asarray(top, dtype=bool)
        random = np.asarray(random, dtype=float) >= 0.5
        only_top = int((top & ~random).sum())
        only_random = int((~top & random).sum())
        if only_top + only_random == 0:
            return 1.0
        return float(binomtest(only_top, only_top + only_random, 0.5).pvalue)

    summary = {"n": len(rows), "k_fraction": K_FRACTION, "modalities": {}}
    for name in ("text", "image", "both"):
        top = np.array([row[f"{name}_flip"] for row in rows], dtype=float)
        rand = np.array([row[f"{name}_random_flip"] for row in rows], dtype=float)
        summary["modalities"][name] = {
            "top_flip_rate": float(top.mean()),
            "random_flip_rate": float(rand.mean()),
            "difference": float((top - rand).mean()),
            "mcnemar_p": exact_mcnemar(top, rand),
            "top_flips": int(top.sum()),
            "n": len(rows),
        }

    classes = {}
    for label in ("correct", "false_positive", "false_negative", "over_trigger", "misclass"):
        subset = [row for row in rows if row["outcome"] == label]
        if not subset:
            classes[label] = {"n": 0}
            continue
        classes[label] = {
            "n": len(subset),
            "share": len(subset) / len(rows),
            "text_flip_rate": float(np.mean([row["text_flip"] for row in subset])),
            "image_flip_rate": float(np.mean([row["image_flip"] for row in subset])),
            "both_flip_rate": float(np.mean([row["both_flip"] for row in subset])),
            "text_random_flip_rate": float(np.mean([row["text_random_flip"] for row in subset])),
            "image_random_flip_rate": float(np.mean([row["image_random_flip"] for row in subset])),
            "both_random_flip_rate": float(np.mean([row["both_random_flip"] for row in subset])),
        }
    summary["outcomes"] = classes

    # does the error class change how often the ablation flips the decision?
    from scipy.stats import chi2_contingency

    groups = [[row["both_flip"] for row in rows if row["outcome"] == label]
              for label in classes if classes[label]["n"] > 20]
    if len(groups) > 1:
        table = np.array([[sum(group), len(group) - sum(group)] for group in groups])
        if table[:, 0].sum() and (table[:, 1].sum()):
            summary["outcome_flip_table"] = {"chi2": float(chi2_contingency(table)[0]),
                                             "p": float(chi2_contingency(table)[1]),
                                             "groups": [label for label in classes if classes[label]["n"] > 20],
                                             "table": table.tolist()}

    with open(os.path.join(OUT, "p4_flip.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[3] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
