"""P5: human agreement --- is text reliance tied to how far annotators agree?

Pre-registered primary analysis: restrict to the hate-relevant subset (at least one
annotator gives a usable hate label; `religion` does not count as a usable label), encode
agreement as 1.0 / 0.67 / 0.33 from the raw annotations, and correlate it with the text
share |phi_T| / (|phi_T| + |phi_I|) by a one-sided (negative) Spearman test. Controls: a
partial correlation that partials out the human non-zero vote count and the v(T,I)
quartile, because both move with agreement and with the target level. Buckets and the
full-sample correlation are reported as descriptives / appendix; with three buckets the
smallest one-sided p at bucket level is 1/6, so a bucket test cannot carry a claim.

Output: refine-logs/p5_agreement.json
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p2_phi.jsonl")
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
USABLE = {1, 2, 3, 5}


def ranks(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.argsort(np.argsort(values)), dtype=np.float64)


def residualise(values: np.ndarray, controls: list) -> np.ndarray:
    design = np.column_stack([np.ones_like(values)] + list(controls))
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list):
    from scipy.stats import spearmanr

    rank_controls = [ranks(control) for control in controls]
    rx = residualise(ranks(x), rank_controls)
    ry = residualise(ranks(y), rank_controls)
    return spearmanr(rx, ry)


def one_sided(result) -> float:
    return float(result.pvalue / 2) if result.statistic < 0 else float(1 - result.pvalue / 2)


def describe(agree: np.ndarray, share: np.ndarray, phi_t: np.ndarray, phi_i: np.ndarray,
             target: np.ndarray) -> dict:
    from scipy.stats import kruskal

    buckets = {}
    for level, name in ((1.0, "all_agree"), (2.0 / 3.0, "partial"), (1.0 / 3.0, "all_differ")):
        mask = np.isclose(agree, level)
        buckets[name] = {
            "n": int(mask.sum()),
            "text_share_mean": float(share[mask].mean()) if mask.any() else None,
            "text_share_median": float(np.median(share[mask])) if mask.any() else None,
            "phi_t_mean": float(phi_t[mask].mean()) if mask.any() else None,
            "phi_i_mean": float(phi_i[mask].mean()) if mask.any() else None,
            "target_mean": float(target[mask].mean()) if mask.any() else None,
        }
    groups = [share[np.isclose(agree, level)] for level in (1.0, 2.0 / 3.0, 1.0 / 3.0)]
    groups = [group for group in groups if group.size]
    kw = kruskal(*groups) if len(groups) > 1 else None
    return {"buckets": buckets,
            "kruskal_statistic": float(kw.statistic) if kw else None,
            "kruskal_p": float(kw.pvalue) if kw else None,
            "kruskal_note": "descriptive only: three groups cannot support a bucket-level claim"}


def main() -> None:
    from scipy.stats import spearmanr

    rows = []
    with open(JSONL, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)

    agree, share, phi_t, phi_i, target, votes, hate = [], [], [], [], [], [], []
    for row in rows:
        labels = [int(label) for label in gt[row["id"]]["labels"]]
        usable_votes = sum(1 for label in labels if label in USABLE)
        agree.append(row["agree"])
        votes.append(usable_votes)
        hate.append(usable_votes >= 1)
        t, i = row["phi_t"], row["phi_i"]
        phi_t.append(t)
        phi_i.append(i)
        share.append(abs(t) / (abs(t) + abs(i) + 1e-12))
        target.append(row["v_both"])

    agree = np.asarray(agree, float)
    share = np.asarray(share, float)
    phi_t = np.asarray(phi_t, float)
    phi_i = np.asarray(phi_i, float)
    target = np.asarray(target, float)
    votes = np.asarray(votes, float)
    hate = np.asarray(hate, bool)

    quartile = np.asarray(np.searchsorted(np.quantile(target, [0.25, 0.5, 0.75]), target), float)

    def block(index: np.ndarray) -> dict:
        x, y = agree[index], share[index]
        raw = spearmanr(x, y)
        partial = partial_spearman(x, y, [votes[index], quartile[index]])
        out = describe(x, y, phi_t[index], phi_i[index], target[index])
        out.update({
            "n": int(index.sum()),
            "spearman_rho": float(raw.statistic),
            "spearman_p_two_sided": float(raw.pvalue),
            "spearman_p_one_sided_negative": one_sided(raw),
            "partial_rho": float(partial.statistic),
            "partial_p_one_sided_negative": one_sided(partial),
            "partial_controls": ["human usable-vote count", "v(T,I) quartile"],
        })
        return out

    primary = block(hate)
    appendix = block(np.ones_like(hate, dtype=bool))

    summary = {
        **primary,
        "criterion": "primary (hate-relevant subset): rho < -0.2 and one-sided p < 0.05",
        "passes": bool(primary["spearman_rho"] < -0.2 and primary["spearman_p_one_sided_negative"] < 0.05),
        "full_sample": appendix,
        "n_excluded": int((~hate).sum()),
    }
    with open(os.path.join(OUT, "p5_agreement.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[P5] primary n={summary['n']} rho={summary['spearman_rho']:+.4f} "
          f"p1={summary['spearman_p_one_sided_negative']:.3e} "
          f"partial={summary['partial_rho']:+.4f} p1={summary['partial_p_one_sided_negative']:.3e}")
    print(f"[P5] full n={appendix['n']} rho={appendix['spearman_rho']:+.4f} "
          f"partial={appendix['partial_rho']:+.4f}")


if __name__ == "__main__":
    main()
