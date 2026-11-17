"""Smoke test the SigLIP2 fusion loader on four samples (P7 prerequisite)."""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from p7_siglip2 import SiglipWrapper, fusion_probs, load_siglip  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
MODEL_DIR = r"C:\Users\123\AppData\Local\Temp\opencode\siglip2_model"


def main() -> None:
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor, config = load_siglip(MODEL_DIR, device)
    print(f"[1] loaded on {device}; thresholds={config.thresholds}")

    with open(os.path.join(OUT, "p2_phi.jsonl"), encoding="utf-8") as handle:
        ids = [json.loads(line)["id"] for line in handle if line.strip()][:4]
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)

    wrapper = SiglipWrapper(model, tokenizer, processor, device)
    texts = [gt[t].get("tweet_text", "") for t in ids]
    images = [Image.open(os.path.join(ROOT, "img_resized", f"{t}.jpg")).convert("RGB") for t in ids]
    tfeat, vfeat = wrapper.features(texts, images)
    print(f"[2] features {tuple(tfeat.shape)} {tuple(vfeat.shape)} "
          f"norms t={tfeat.norm(dim=1).mean():.3f} v={vfeat.norm(dim=1).mean():.3f}")

    with torch.no_grad():
        probs = fusion_probs(model, tfeat.to(device), vfeat.to(device))
        print(f"[3] fusion probs {tuple(probs.shape)}")
        for row in range(len(ids)):
            print("    ", [round(float(x), 3) for x in probs[row]])

        blank = wrapper.encode_text([""])
        tblank = wrapper.text_features(blank).cpu()
        vblank = model.backbone.get_image_features(
            pixel_values=torch.zeros(1, 3, 224, 224, device=device)
        ).pooler_output.cpu()
        t = tfeat[:2].to(device)
        v = vfeat[:2].to(device)
        tb = tblank.to(device).expand_as(t)
        vb = vblank.to(device).expand_as(v)
        v_empty = fusion_probs(model, tb, vb)
        v_both = fusion_probs(model, t, v)
        print(f"[4] blank text feature norm {tblank.norm():.3f}  "
              f"v_empty {v_empty.mean():.4f} -> v_both {v_both.mean():.4f}")


if __name__ == "__main__":
    main()
