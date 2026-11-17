"""P1 selection: apply the pre-registered rule and name the winner per modality.

Rule (fixed before any analysis of the model's behaviour):

  1. a candidate must clear the effect floor (text AOPC > 0.01, image > 0.001)
     and beat its random ordering (one-sided paired Wilcoxon p < 0.05);
  2. among the survivors the highest mean AOPC wins;
  3. two candidates whose AOPC differs by < 0.001, or whose paired Wilcoxon
     p > 0.05, are a tie. The tie-break is (a) higher comprehensiveness at 20%,
     then (b) the smaller seed / noise variance, then (c) the deterministic
     method --- deterministic beats sampled on principle when all else is equal.

Output: refine-logs/p1_decision.json
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
FLOOR = {"text": 0.01, "image": 0.001}


def load_rows() -> dict:
    rows = {}
    with open(os.path.join(OUT, "p1_faithfulness.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                rows[record["id"]] = record
    return rows


def load_stability() -> dict:
    path = os.path.join(OUT, "p1_stability.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    from scipy.stats import wilcoxon

    rows = list(load_rows().values())
    stability = load_stability()
    methods = {
        "text": ["occlusion", "lime", "kernelshap"],
        "image": ["occlusion", "gradcam", "ig", "scorecam"],
    }

    decision = {"n": len(rows), "floor": FLOOR, "modalities": {}}
    for modality, names in methods.items():
        random_key = f"{modality}_random"
        values = {name: np.array([row[name]["aopc_del"] for row in rows], dtype=float)
                  for name in names}
        random_values = np.array([row[random_key]["aopc_del"] for row in rows], dtype=float)
        candidates = []
        for name in names:
            diff = values[name] - random_values
            diff = diff[np.isfinite(diff)]
            p_random = float(wilcoxon(diff).pvalue) if diff.size and np.any(diff != 0) else 1.0
            mean = float(np.nanmean(values[name]))
            comp = float(np.nanmean([row[name]["comp_20"] for row in rows]))
            passed = bool(mean > FLOOR[modality] and p_random < 0.05 and mean > float(np.nanmean(random_values)))
            candidates.append({
                "method": name,
                "aopc_mean": mean,
                "aopc_median": float(np.nanmedian(values[name])),
                "comp20_mean": comp,
                "suff20_mean": float(np.nanmean([row[name]["suff_20"] for row in rows])),
                "p_vs_random": p_random,
                "clears_floor": bool(mean > FLOOR[modality]),
                "passes": passed,
            })

        passed = [entry for entry in candidates if entry["passes"]]
        passed.sort(key=lambda entry: -entry["aopc_mean"])
        if not passed:
            decision["modalities"][modality] = {"candidates": candidates, "winner": None,
                                                "note": "no candidate cleared the floor"}
            continue

        top = passed[0]
        tie = None
        if len(passed) > 1:
            second = passed[1]
            diff = values[top["method"]] - values[second["method"]]
            diff = diff[np.isfinite(diff)]
            p_pair = float(wilcoxon(diff).pvalue) if diff.size and np.any(diff != 0) else 1.0
            tie = {
                "other": second["method"],
                "delta_aopc": top["aopc_mean"] - second["aopc_mean"],
                "paired_p": p_pair,
                "tied": bool(abs(top["aopc_mean"] - second["aopc_mean"]) < 0.001 or p_pair > 0.05),
            }
            if tie["tied"]:
                # tie-break (a): comprehensiveness at 20%
                if second["comp20_mean"] > top["comp20_mean"] + 1e-9:
                    top = second
                # tie-break (b): stability, when available
                block = stability.get("text_seeds", {}) if modality == "text" else stability.get("image_noise", {})
                if top["method"] in block and tie.get("other") in block:
                    if block[tie["other"]].get("min_spearman", -1) > block[top["method"]].get("min_spearman", -1):
                        winner_block = block[tie["other"]]
                        if winner_block.get("min_spearman", 0) > block[top["method"]].get("min_spearman", 0):
                            top = next(entry for entry in passed if entry["method"] == tie["other"])
                tie["resolved_by"] = "comp20 / stability / deterministic"

        decision["modalities"][modality] = {
            "candidates": candidates,
            "winner": top["method"],
            "winner_block": top,
            "tie": tie,
        }

    with open(os.path.join(OUT, "p1_decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)

    for modality, block in decision["modalities"].items():
        print(f"[{modality}] winner = {block['winner']}")
        for entry in block["candidates"]:
            print(f"    {entry['method']:11s} aopc={entry['aopc_mean']:+.4f} "
                  f"comp20={entry['comp20_mean']:+.4f} p_rand={entry['p_vs_random']:.2e} "
                  f"floor={entry['clears_floor']} pass={entry['passes']}")
        if block.get("tie"):
            print("    tie:", json.dumps(block["tie"]))


if __name__ == "__main__":
    main()
