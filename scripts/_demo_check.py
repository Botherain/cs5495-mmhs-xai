"""Draft of the notebook's live demonstration cell (checked as a script first)."""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import (  # noqa: E402
    GRID, content_positions, encode_features, image_occlusion, target_of,
    text_kernelshap, text_lime, text_occlusion, fusion_probs,
)
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")


def main() -> None:
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model", device=device
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["selection"]

    picked = None
    for tweet_id in ids:
        entry = gt[str(tweet_id)]
        if len(entry.get("tweet_text", "")) > 60:
            picked = str(tweet_id)
            break
    entry = gt[picked]
    sample = Sample(picked, entry["tweet_text"], "",
                    os.path.join(ROOT, "img_resized", f"{picked}.jpg"),
                    normalize_labels(entry["labels"]))
    image = Image.open(sample.image_path).convert("RGB")

    encoded = wrapper.encode_text([sample.text])
    ids_row, mask_row = encoded["input_ids"][0], encoded["attention_mask"][0]
    positions = content_positions(mask_row)
    pixel = wrapper.encode_image([image])["pixel_values"]
    tfeat, vfeat = encode_features(wrapper, [sample.text], [image])
    with torch.no_grad():
        probs = fusion_probs(model, tfeat.to(device), vfeat.to(device))[0].cpu()
    tokens = tokenizer.convert_ids_to_tokens(
        ids_row[positions].tolist()
    )

    text_values = {
        "occlusion": text_occlusion(wrapper, ids_row, mask_row, positions, vfeat[0]),
        "lime": text_lime(wrapper, ids_row, mask_row, positions, vfeat[0]),
        "kernelshap": text_kernelshap(wrapper, ids_row, mask_row, positions, vfeat[0]),
    }
    grid = image_occlusion(wrapper, pixel, tfeat)[0]
    print("id:", picked)
    print("text:", sample.text[:120])
    print("target:", round(float(target_of(probs)), 4),
          "heads:", [round(float(x), 3) for x in probs])
    for name, values in text_values.items():
        order = np.argsort(-values)[:6]
        print(f"  {name:11s}", [(tokens[i], round(float(values[i]), 4)) for i in order])

    order = np.argsort(-grid)[:6]
    print("  image cells:", [(divmod(int(i), GRID), round(float(grid[i]), 5)) for i in order])

    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[0].axis("off")
    heat = grid.reshape(GRID, GRID)
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-12)
    axes[1].imshow(image)
    axes[1].imshow(np.kron(heat, np.ones((32, 32))), cmap="inferno", alpha=0.55)
    axes[1].set_title("occlusion attribution (7x7)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "demo_sample.png"), dpi=130)
    print("figure written")


if __name__ == "__main__":
    main()
