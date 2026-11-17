"""Report figures from the stored result files (no model, no GPU).

  figures/fig1_aopc.png      P1 faithfulness per candidate method vs random
  figures/fig2_phi.png       P2 per-sample phi_T / phi_I distributions
  figures/fig3_deletion.png  P2 deletion curves, text / image / random
  figures/fig4_gates.png     P1.4 tower-randomisation gate curves (if present)

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "refine-logs")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_json(name: str):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fig_aopc() -> None:
    matrix = load_json("p1_matrix.json")
    if not matrix:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for axis, modality in zip(axes, ("text", "image")):
        block = matrix[modality]
        names = list(block)
        values = [block[name]["aopc_del"]["mean"] for name in names]
        random_mean = block[names[0]]["aopc_del"]["random_mean"]
        errors = [block[name]["aopc_del"]["std"] / np.sqrt(block[name]["aopc_del"]["n"])
                  for name in names]
        colours = ["#2b6cb0" if value > random_mean else "#c05621" for value in values]
        axis.bar(range(len(names)), values, yerr=errors, color=colours, width=0.62, capsize=2)
        axis.axhline(random_mean, color="0.35", linestyle="--", linewidth=1)
        axis.axhline(0, color="0.7", linewidth=0.8)
        axis.set_xticks(range(len(names)), names, rotation=20, ha="right")
        axis.set_ylabel("AOPC (deletion + insertion) / 2")
        axis.set_title(f"{modality}  (dashed = random order)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_aopc.png"), dpi=200)
    plt.close(fig)


def fig_phi() -> None:
    summary = load_json("p2_summary.json")
    if not summary:
        return
    phi_t, phi_i = [], []
    with open(os.path.join(OUT, "p2_phi.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                phi_t.append(record["phi_t"])
                phi_i.append(record["phi_i"])
    phi_t = np.asarray(phi_t)
    phi_i = np.asarray(phi_i)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    bins = np.linspace(-0.25, 0.35, 61)
    axes[0].hist(phi_t, bins=bins, color="#2b6cb0", alpha=0.85)
    axes[0].axvline(0, color="k", linewidth=0.8)
    axes[0].axvline(phi_t.mean(), color="#c53030", linewidth=1.2,
                    label=f"mean {phi_t.mean():+.4f}")
    axes[0].set_title(r"$\phi_T$ (text)", fontsize=9)
    axes[0].set_xlabel("Shapley value of the text modality")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(np.clip(phi_i, -0.01, 0.02), bins=np.linspace(-0.01, 0.02, 61),
                 color="#276749", alpha=0.85)
    axes[1].axvline(0, color="k", linewidth=0.8)
    axes[1].axvline(phi_i.mean(), color="#c53030", linewidth=1.2,
                    label=f"mean {phi_i.mean():+.4f}")
    axes[1].set_title(r"$\phi_I$ (image)", fontsize=9)
    axes[1].set_xlabel("Shapley value of the image modality")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_phi.png"), dpi=200)
    plt.close(fig)


def fig_deletion() -> None:
    curves = load_json("p2_curves.json")
    if not curves:
        return
    alphas = np.asarray(curves["alphas"])
    text = np.asarray(curves["text"])
    image = np.asarray(curves["image"])
    random = np.asarray(curves["random"])

    fig, axis = plt.subplots(figsize=(4.2, 3.0))
    for values, label, colour in (
        (text, "text tokens", "#2b6cb0"),
        (image, "image cells", "#276749"),
        (random[:, :, 0] if random.ndim == 3 else random, "random order", "0.45"),
    ):
        mean = values.mean(axis=0)
        low, high = np.percentile(values, [2.5, 97.5], axis=0)
        axis.plot(alphas, mean, marker="o", markersize=3, color=colour, label=label)
        axis.fill_between(alphas, low, high, color=colour, alpha=0.15, linewidth=0)
    axis.set_xlabel("fraction of units deleted")
    axis.set_ylabel("drop in target probability")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_deletion.png"), dpi=200)
    plt.close(fig)


def fig_gates() -> None:
    payload = load_json("p1_sanity_curves.json")
    if not payload:
        return
    curves = payload["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    for axis, modality, names in zip(axes, ("text", "image"),
                                     (("occlusion", "lime", "kernelshap"),
                                      ("occlusion", "gradcam", "ig", "scorecam"))):
        levels = sorted(int(level) for level in curves.get(modality, {}))
        if not levels:
            continue
        for name in names:
            series = [curves[modality][str(level)][name]["spearman"] for level in levels]
            axis.plot(levels, series, marker="o", markersize=3, label=name)
        axis.set_title(f"{modality} tower", fontsize=9)
        axis.set_xlabel("encoder blocks randomised")
        axis.set_xticks(levels)
        axis.axhline(0, color="0.8", linewidth=0.7)
    axes[0].set_ylabel(r"Spearman($A_d$, $A_0$)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_gates.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig_aopc()
    fig_phi()
    fig_deletion()
    fig_gates()
    for name in sorted(os.listdir(FIG)):
        print("wrote", name)


if __name__ == "__main__":
    main()
