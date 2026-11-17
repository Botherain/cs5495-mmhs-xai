"""Smoke test for the P1 attribution toolkit on a handful of samples."""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import (  # noqa: E402
    IMAGE_METHODS, content_positions, encode_features, fusion_probs, image_gradcam,
    image_ig, image_occlusion, image_scorecam, target_of, text_kernelshap,
    text_lime, text_occlusion,
)
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")


def main() -> None:
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["main"][:3]

    samples, images = [], []
    for tweet_id in ids:
        entry = gt[str(tweet_id)]
        samples.append(Sample(str(tweet_id), entry["tweet_text"], "",
                              os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg"),
                              normalize_labels(entry["labels"])))
        images.append(Image.open(samples[-1].image_path).convert("RGB"))

    print("[1] texts:")
    for sample in samples:
        print("    ", sample.text[:90].replace("\n", " "))

    tfeat, vfeat = encode_features(wrapper, [s.text for s in samples], images)
    encoded = wrapper.encode_text([s.text for s in samples])
    encoded_img = wrapper.encode_image(images)

    t0 = time.time()
    probs_check = fusion_probs(model, tfeat.to(device), vfeat.to(device))
    probs_ref = wrapper.head_probs(encoded["input_ids"], encoded["attention_mask"],
                                   encoded_img["pixel_values"])
    print(f"[2] fusion parity max|diff| = {(probs_check.cpu() - probs_ref).abs().max():.3e} ({time.time() - t0:.2f}s)")
    print("    target =", target_of(probs_ref).tolist())

    for index in range(len(samples)):
        ids_row = encoded["input_ids"][index]
        mask_row = encoded["attention_mask"][index]
        positions = content_positions(mask_row)
        fixed_v = vfeat[index]
        print(f"[3] sample {index}: tokens={len(positions)}")

        for name, fn in (
            ("occlusion", text_occlusion),
            ("lime", text_lime),
            ("kernelshap", text_kernelshap),
        ):
            started = time.time()
            values = fn(wrapper, ids_row, mask_row, positions, fixed_v)
            print(f"    {name:11s} {time.time() - started:6.2f}s  sum={values.sum():+.4f} "
                  f"min={values.min():+.4f} max={values.max():+.4f}")

        pixel = encoded_img["pixel_values"][index]
        fixed_t = tfeat[index]
        for name, fn in (
            ("occlusion", image_occlusion),
            ("gradcam", image_gradcam),
            ("ig", image_ig),
            ("scorecam", image_scorecam),
        ):
            started = time.time()
            values = fn(wrapper, pixel, fixed_t)
            print(f"    {name:11s} {time.time() - started:6.2f}s  shape={values.shape} "
                  f"sum={values.sum():+.4f} min={values.min():+.4f} max={values.max():+.4f}")

    with torch.no_grad():
        pixel = encoded_img["pixel_values"][0]
        full = target_of(fusion_probs(model, tfeat[0].to(device), vfeat[0].to(device)
                                      .unsqueeze(0))).item()
        blank_t = target_of(fusion_probs(model, torch.zeros_like(tfeat[0].to(device)).unsqueeze(0),
                                         vfeat[0].to(device).unsqueeze(0))).item()
        print(f"[4] v(full)={full:.6f} v(blank text)={blank_t:.6f} "
              f"image methods listed: {IMAGE_METHODS}")


if __name__ == "__main__":
    main()
