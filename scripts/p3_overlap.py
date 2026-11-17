"""P3.1: where does the image attribution land --- on printed text or elsewhere?

Uses the winner's image attribution on the analysis set (P1.7, 7x7 occlusion
cells). A cell's text weight is the fraction of its area covered by EasyOCR
boxes, so a cell that a small box only partly covers still counts, and no sample
is dropped for having text too small to contain a cell centre.

The null keeps the total text area fixed and moves a square of that area to a
random position (100 placements), weighting cells by exact overlap in the same
way. The comparison is therefore "the same amount of image, placed on the text"
against "placed elsewhere", not against zero.

Outputs
  refine-logs/p3_overlap.jsonl   per sample
  refine-logs/p3_overlap.json    aggregates + paired test

Usage: python scripts/p3_overlap.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import GRID  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p3_overlap.jsonl")
N_NULL = 100
SIDE = 224.0
CELL = SIDE / GRID


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


def rect_weights(x1: float, x2: float, y1: float, y2: float) -> np.ndarray:
    """Fraction of each 7x7 cell covered by an axis-aligned rectangle."""
    edges = np.arange(GRID) * CELL
    wx = np.clip(np.minimum(x2, edges + CELL) - np.maximum(x1, edges), 0.0, CELL) / CELL
    wy = np.clip(np.minimum(y2, edges + CELL) - np.maximum(y1, edges), 0.0, CELL) / CELL
    return np.outer(wy, wx).reshape(GRID * GRID)


def share(energy: np.ndarray, weights: np.ndarray) -> float:
    total = energy.sum()
    if total <= 0:
        return float("nan")
    return float((energy * weights).sum() / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ocr = load_jsonl(os.path.join(OUT, "p0_ocr.jsonl"))
    attributions = load_jsonl(os.path.join(OUT, "p1d_image_attr_test.jsonl"))
    finished = set(load_jsonl(JSONL))
    ids = [tweet_id for tweet_id in attributions if tweet_id not in finished]
    if args.limit:
        ids = ids[:args.limit]
    print(f"[1] attributions={len(attributions)} ocr={len(ocr)} todo={len(ids)}", flush=True)
    if not ids:
        summarise()
        return

    rng = np.random.default_rng(0)
    with open(JSONL, "a", encoding="utf-8") as sink:
        for step, tweet_id in enumerate(ids, start=1):
            record = ocr.get(tweet_id, {})
            width = float(record.get("width") or 224)
            height = float(record.get("height") or 224)
            boxes = record.get("boxes") or []
            scale_x = SIDE / max(width, 1.0)
            scale_y = SIDE / max(height, 1.0)

            keep = 1.0 - np.zeros(GRID * GRID)
            area = 0.0
            for box in boxes:
                x1, x2 = box["x1"] * scale_x, box["x2"] * scale_x
                y1, y2 = box["y1"] * scale_y, box["y2"] * scale_y
                area += max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
                keep = keep * (1.0 - rect_weights(x1, x2, y1, y2))
            weights = 1.0 - keep

            energy = np.abs(np.asarray(
                attributions[tweet_id].get("values", attributions[tweet_id].get("occlusion")),
                dtype=np.float64,
            ))
            observed = share(energy, weights) if boxes else float("nan")

            # The control must match the *union* of the text region, not the sum of
            # the box areas: overlapping boxes make the sum larger than the region
            # the observed statistic actually covers, which would inflate the control
            # and push the excess negative for purely geometric reasons.
            union_area = min(float(weights.sum()) * CELL * CELL, SIDE * SIDE)
            sum_area = min(area, SIDE * SIDE)
            null_union, null_sum = [], []
            for target_area, sink_list in ((union_area, null_union), (sum_area, null_sum)):
                side = float(np.sqrt(target_area))
                if side <= 0:
                    continue
                for _ in range(N_NULL):
                    left = rng.uniform(0.0, max(SIDE - side, 0.0))
                    top = rng.uniform(0.0, max(SIDE - side, 0.0))
                    sink_list.append(share(energy, rect_weights(left, left + side, top, top + side)))

            sink.write(json.dumps({
                "id": tweet_id,
                "has_text": bool(boxes),
                "n_boxes": len(boxes),
                "area_share": union_area / (SIDE * SIDE),
                "area_sum_share": sum_area / (SIDE * SIDE),
                "cells_touched": int((weights > 0).sum()),
                "text_weight_mass": float(weights.sum()),
                "observed_share": observed,
                "null_mean": float(np.mean(null_union)) if null_union else None,
                "null_p95": float(np.percentile(null_union, 95)) if null_union else None,
                "null_mean_sum_area": float(np.mean(null_sum)) if null_sum else None,
                "total_energy": float(energy.sum()),
            }) + "\n")
            if step % 250 == 0 or step == len(ids):
                sink.flush()
                print(f"    {step}/{len(ids)}", flush=True)

    summarise()


def summarise() -> None:
    from scipy.stats import wilcoxon

    rows = list(load_jsonl(JSONL).values())
    with_text = [row for row in rows if row["has_text"] and row["null_mean"] is not None
                 and np.isfinite(row["observed_share"])]
    if not with_text:
        print("[2] no usable rows")
        return
    observed = np.array([row["observed_share"] for row in with_text])
    null = np.array([row["null_mean"] for row in with_text])
    null_sum = np.array([row["null_mean_sum_area"] for row in with_text
                         if row.get("null_mean_sum_area") is not None])
    excess = observed - null
    p_value = float(wilcoxon(excess).pvalue) if np.any(excess != 0) else None
    summary = {
        "n_total": len(rows),
        "n_with_text": len(with_text),
        "mean_observed_share": float(observed.mean()),
        "mean_null_share": float(null.mean()),
        "mean_excess": float(excess.mean()),
        "median_excess": float(np.median(excess)),
        "excess_ci95": [float(np.percentile(excess, 2.5)), float(np.percentile(excess, 97.5))],
        "p_wilcoxon": p_value,
        "mean_null_sum_area_share": float(null_sum.mean()) if null_sum.size else None,
        "mean_excess_sum_area": float((observed - null_sum).mean()) if null_sum.size else None,
        "share_above_null_p95": float(np.mean(observed > np.array([row["null_p95"] for row in with_text]))),
        "mean_text_area_share": float(np.mean([row["area_share"] for row in with_text])),
        "mean_cells_touched": float(np.mean([row["cells_touched"] for row in with_text])),
        "criterion": "excess > 0.15 and paired p < 0.05",
        "passes": bool(excess.mean() > 0.15 and p_value is not None and p_value < 0.05),
        "null_placements": N_NULL,
    }
    with open(os.path.join(OUT, "p3_overlap.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[2] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
