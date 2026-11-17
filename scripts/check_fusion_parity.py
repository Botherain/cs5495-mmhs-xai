"""Parity check for the CLIP fusion path used everywhere in the report.

`fusion_probs` reimplements the checkpoint's forward from the two pooler tensors so the
coalition game and every attribution can run without the towers. If that reimplementation
disagrees with `Wrapper.head_probs` (which goes through the real forward), every number in
§2-§8 inherits the disagreement. This checks eight real samples.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import encode_features, fusion_probs  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")


def main() -> None:
    from PIL import Image

    import json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["main"][:8]
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    texts = [gt[tweet_id].get("tweet_text", "") for tweet_id in ids]
    images = [Image.open(os.path.join(ROOT, "img_resized", f"{t}.jpg")).convert("RGB") for t in ids]

    encoded = wrapper.encode_text(texts)
    pixel = wrapper.encode_image(images)["pixel_values"]
    with torch.no_grad():
        forward = wrapper.head_probs(encoded["input_ids"], encoded["attention_mask"], pixel)
        tfeat, vfeat = encode_features(wrapper, texts, images)
        reimpl = fusion_probs(model, tfeat.to(device), vfeat.to(device))

    delta = (forward.cpu() - reimpl.cpu()).abs()
    print(f"max |delta| = {delta.max().item():.3e}")
    print(f"mean |delta| = {delta.mean().item():.3e}")
    if delta.max().item() > 1e-5:
        raise SystemExit("PARITY FAILED")
    print("parity ok")


if __name__ == "__main__":
    main()
