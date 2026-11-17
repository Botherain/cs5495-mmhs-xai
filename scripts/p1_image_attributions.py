"""P1.2: image-side attributions on the 499-sample selection set.

Four candidates, one unit (7x7 cell of the vision tower's own patch grid):

  occlusion   leave-one-patch-out (mean colour)
  gradcam     Grad-CAM on the last vision block
  ig          integrated gradients from the mean colour, 32 steps
  scorecam    Score-CAM with 7x7 upsampled masks

Samples are batched: towers are independent across the batch, so one backward of
the summed target yields each sample its own gradient.

Output: refine-logs/p1_image_attr.jsonl (one line per sample, resumable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import (  # noqa: E402
    IMAGE_METHODS, encode_features, image_gradcam, image_ig, image_occlusion,
    image_scorecam,
)
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p1_image_attr.jsonl")


def load_samples(limit: int = 0) -> list:
    from PIL import Image

    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["selection"]
    samples, images = [], []
    for tweet_id in ids:
        path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(path):
            continue
        entry = gt[str(tweet_id)]
        samples.append(
            Sample(str(tweet_id), entry.get("tweet_text", ""), "", path, normalize_labels(entry["labels"]))
        )
        images.append(Image.open(path).convert("RGB"))
    if limit:
        samples, images = samples[:limit], images[:limit]
    return samples, images


def done_ids() -> set:
    if not os.path.exists(JSONL):
        return set()
    done = set()
    with open(JSONL, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    samples, images = load_samples(args.limit)
    finished = done_ids()
    todo = [index for index, sample in enumerate(samples) if sample.tweet_id not in finished]
    print(f"[1] samples={len(samples)} done={len(finished)} todo={len(todo)}", flush=True)
    if not todo:
        return

    encoded_img = wrapper.encode_image(images)
    pixels = encoded_img["pixel_values"]
    tfeat, _ = encode_features(wrapper, [s.text for s in samples], images)

    os.makedirs(OUT, exist_ok=True)
    started = time.time()
    with open(JSONL, "a", encoding="utf-8") as sink:
        for step, start in enumerate(range(0, len(todo), args.batch), start=1):
            chunk = todo[start:start + args.batch]
            pixel = pixels[chunk].to(device)
            text = tfeat[chunk].to(device)
            outputs = {
                "occlusion": image_occlusion(wrapper, pixel, text),
                "gradcam": image_gradcam(wrapper, pixel, text),
                "ig": image_ig(wrapper, pixel, text),
                "scorecam": image_scorecam(wrapper, pixel, text),
            }
            for row, index in enumerate(chunk):
                record = {"id": samples[index].tweet_id}
                for name in IMAGE_METHODS:
                    record[name] = [float(value) for value in outputs[name][row]]
                sink.write(json.dumps(record) + "\n")
            sink.flush()
            done = start + len(chunk)
            if step % 5 == 0 or done >= len(todo):
                rate = done / max(time.time() - started, 1e-6)
                print(
                    f"    {done}/{len(todo)} {rate:.2f}/s eta {(len(todo) - done) / rate / 60:.1f} min",
                    flush=True,
                )

    print("[2] done", flush=True)


if __name__ == "__main__":
    main()
