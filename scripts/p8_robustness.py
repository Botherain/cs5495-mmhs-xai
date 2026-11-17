"""P8: robustness checks that could invalidate the main claim.

  A3  preprocessing ablation: strip URLs, @mentions and the trailing link from the
      tweet, recompute the four coalitions, and compare phi with the raw text. The
      criterion is a distribution shift below 10%, so "the conclusion is an
      artefact of the truncation / preprocessing" is excluded before it is raised.
  target sensitivity: the analysis target is the mean of four usable heads; the
      five-head mean is recomputed on the same features to check that the ordering
      does not depend on dropping religion.

Output: refine-logs/p8_robustness.json
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import USABLE_HEADS, encode_features, fusion_probs  # noqa: E402
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")


def strip_markup(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\bRT\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def coalitions(model, tfeat, vfeat, tblank, vblank) -> tuple:
    device = tfeat.device
    tb = tblank.to(device).expand_as(tfeat)
    vb = vblank.to(device).expand_as(vfeat)
    with torch.no_grad():
        v_empty = fusion_probs(model, tb, vb)
        v_text = fusion_probs(model, tfeat, vb)
        v_image = fusion_probs(model, tb, vfeat)
        v_both = fusion_probs(model, tfeat, vfeat)
    phi_t = 0.5 * (v_text - v_empty) + 0.5 * (v_both - v_image)
    phi_i = 0.5 * (v_image - v_empty) + 0.5 * (v_both - v_text)
    return v_empty, v_text, v_image, v_both, phi_t, phi_i


def main() -> None:
    from PIL import Image
    from scipy.stats import spearmanr

    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        subset = json.load(handle)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    with torch.no_grad():
        blank_ids, blank_mask = wrapper.blank_text(1)
        tblank = model._encode_text(blank_ids.to(device), blank_mask.to(device)).cpu().squeeze(0)
        vblank = model._encode_image(torch.zeros(1, 3, 224, 224, device=device)).cpu().squeeze(0)

# A3 preprocessing ablation
    selection = [
        Sample(str(t), gt[str(t)].get("tweet_text", ""), "",
               os.path.join(ROOT, "img_resized", f"{t}.jpg"), normalize_labels(gt[str(t)]["labels"]))
        for t in subset["selection"]
        if os.path.exists(os.path.join(ROOT, "img_resized", f"{t}.jpg"))
    ]
    images = [Image.open(sample.image_path).convert("RGB") for sample in selection]
    raw_t, vfeat = encode_features(wrapper, [s.text for s in selection], images)
    clean_t, _ = encode_features(wrapper, [strip_markup(s.text) for s in selection], images)

    raw = coalitions(model, raw_t.to(device), vfeat.to(device), tblank, vblank)
    clean = coalitions(model, clean_t.to(device), vfeat.to(device), tblank, vblank)
    # Shapley values are linear in the target, so averaging the per-head values
    # over the four usable heads is exactly the value of the mean-target game.
    usable = list(USABLE_HEADS)
    phi_t_raw = raw[4][:, usable].mean(dim=1).cpu().numpy()
    phi_i_raw = raw[5][:, usable].mean(dim=1).cpu().numpy()
    phi_t_clean = clean[4][:, usable].mean(dim=1).cpu().numpy()
    phi_i_clean = clean[5][:, usable].mean(dim=1).cpu().numpy()

    def share(phi_t, phi_i):
        return np.abs(phi_t) / (np.abs(phi_t) + np.abs(phi_i) + 1e-12)

    rho_t = spearmanr(phi_t_raw, phi_t_clean).statistic
    shift = float(abs(phi_t_raw.mean() - phi_t_clean.mean()) / (abs(phi_t_raw.mean()) + 1e-12))
    dominance_raw = float((np.abs(phi_t_raw) > np.abs(phi_i_raw)).mean())
    dominance_clean = float((np.abs(phi_t_clean) > np.abs(phi_i_clean)).mean())
    a3 = {
        "n": len(selection),
        "phi_t_mean_raw": float(phi_t_raw.mean()),
        "phi_t_mean_stripped": float(phi_t_clean.mean()),
        "relative_shift": shift,
        "spearman_phi_t": float(rho_t),
        "text_dominance_raw": dominance_raw,
        "text_dominance_stripped": dominance_clean,
        "criterion": "relative shift < 0.10 and dominance ordering unchanged",
        "passes": bool(shift < 0.10 and abs(dominance_raw - dominance_clean) < 0.05),
    }
    print(f"[1] A3 shift={shift:.4f} rho={rho_t:.3f} dominance {dominance_raw:.3f} -> {dominance_clean:.3f}",
          flush=True)

# target sensitivity: four vs five heads
    with open(os.path.join(OUT, "p2_phi.jsonl"), encoding="utf-8") as handle:
        ids = [json.loads(line)["id"] for line in handle if line.strip()]
    texts = [gt[tweet_id].get("tweet_text", "") for tweet_id in ids]
    main_images = [Image.open(os.path.join(ROOT, "img_resized", f"{t}.jpg")).convert("RGB") for t in ids]
    tfeat, vfeat_main = encode_features(wrapper, texts, main_images)
    print(f"[2] analysis features {tfeat.shape}", flush=True)

    summary = {}
    for name, heads in (("four_heads", list(USABLE_HEADS)), ("five_heads", [0, 1, 2, 3, 4])):
        phi_t_all, phi_i_all = [], []
        for start in range(0, len(ids), 256):
            stop = min(start + 256, len(ids))
            block = coalitions(model, tfeat[start:stop].to(device), vfeat_main[start:stop].to(device),
                               tblank, vblank)
            phi_t_all.append(block[4][:, heads].mean(dim=1).cpu().numpy())
            phi_i_all.append(block[5][:, heads].mean(dim=1).cpu().numpy())
        phi_t_all = np.concatenate(phi_t_all)
        phi_i_all = np.concatenate(phi_i_all)
        summary[name] = {
            "phi_t_mean": float(phi_t_all.mean()),
            "phi_i_mean": float(phi_i_all.mean()),
            "text_dominance": float((np.abs(phi_t_all) > np.abs(phi_i_all)).mean()),
            "share_mean": float(share(phi_t_all, phi_i_all).mean()),
        }
        print(f"    {name}: phi_T={summary[name]['phi_t_mean']:+.4f} phi_I={summary[name]['phi_i_mean']:+.4f} "
              f"dominance={summary[name]['text_dominance']:.4f}", flush=True)

    sensitivity = {
        "four_heads": summary["four_heads"],
        "five_heads": summary["five_heads"],
        "dominance_difference": abs(summary["four_heads"]["text_dominance"]
                                    - summary["five_heads"]["text_dominance"]),
        "criterion": "ordering unchanged (dominance difference < 0.05)",
        "passes": bool(abs(summary["four_heads"]["text_dominance"]
                           - summary["five_heads"]["text_dominance"]) < 0.05
                       and summary["five_heads"]["phi_t_mean"] > summary["five_heads"]["phi_i_mean"]),
    }

    payload = {"a3_preprocessing": a3, "target_sensitivity": sensitivity}
    with open(os.path.join(OUT, "p8_robustness.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("[3] " + json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
