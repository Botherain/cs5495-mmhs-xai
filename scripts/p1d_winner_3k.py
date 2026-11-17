"""P1.7: the P1 winner's attribution on the full 3,000-sample analysis set.

P3 and P4 need unit-level attributions for the analysis set, so this script
computes them once and writes them to disk:

  refine-logs/p1d_text_attr_test.jsonl    text winner (LIME), content tokens
  refine-logs/p1d_image_attr_test.jsonl   image winner (occlusion), 7x7 cells

The winner is read from p1_decision.json rather than hard-coded, so the file
names and the methods stay in step with the pre-registered selection rule.

Records carry {"id", "method", "positions", "values"}; the image file has no
"positions" because its units are the fixed 7x7 grid.

Usage:
  python scripts/p1d_winner_3k.py --modalities image
  python scripts/p1d_winner_3k.py --modalities text
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
    content_positions, encode_features, image_occlusion, text_kernelshap,
    text_lime, text_occlusion,
)
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
TEXT_JSONL = os.path.join(OUT, "p1d_text_attr_test.jsonl")
IMAGE_JSONL = os.path.join(OUT, "p1d_image_attr_test.jsonl")

TEXT_METHODS = {"occlusion": text_occlusion, "lime": text_lime, "kernelshap": text_kernelshap}


def load_samples(limit: int = 0) -> list:
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["main"]
    samples = []
    for tweet_id in ids:
        path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(path):
            continue
        entry = gt[str(tweet_id)]
        samples.append(
            Sample(str(tweet_id), entry.get("tweet_text", ""), "", path, normalize_labels(entry["labels"]))
        )
    return samples[:limit] if limit else samples


def done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def winners() -> dict:
    path = os.path.join(OUT, "p1_decision.json")
    default = {"text": "lime", "image": "occlusion"}
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        decision = json.load(handle)
    return {modality: (decision["modalities"].get(modality) or {}).get("winner") or default[modality]
            for modality in ("text", "image")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--modalities", default="text,image")
    parser.add_argument("--text-method", default="", help="override the P1 text winner")
    args = parser.parse_args()
    want = {item.strip() for item in args.modalities.split(",") if item.strip()}

    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    chosen = winners()
    if args.text_method:
        chosen["text"] = args.text_method
    text_method = TEXT_METHODS[chosen["text"]]

    samples = load_samples(args.limit)
    done_text = done_ids(TEXT_JSONL)
    done_image = done_ids(IMAGE_JSONL)
    print(f"[1] samples={len(samples)} text={chosen['text']} image={chosen['image']} "
          f"text_done={len(done_text)} image_done={len(done_image)}", flush=True)

    todo = [index for index, sample in enumerate(samples) if sample.tweet_id not in done_text]
    if todo and "text" in want:
        started = time.time()
        with open(TEXT_JSONL, "a", encoding="utf-8") as sink:
            for begin in range(0, len(todo), args.chunk):
                chunk = todo[begin:begin + args.chunk]
                images = [Image.open(samples[index].image_path).convert("RGB") for index in chunk]
                _, vfeat = encode_features(wrapper, [samples[index].text for index in chunk], images)
                encoded = wrapper.encode_text([samples[index].text for index in chunk])
                for row, index in enumerate(chunk):
                    ids_row = encoded["input_ids"][row]
                    mask_row = encoded["attention_mask"][row]
                    positions = content_positions(mask_row)
                    values = text_method(wrapper, ids_row, mask_row, positions, vfeat[row])
                    sink.write(json.dumps({
                        "id": samples[index].tweet_id,
                        "method": chosen["text"],
                        "n_tokens": len(positions),
                        "positions": positions,
                        "values": [float(value) for value in values],
                    }) + "\n")
                sink.flush()
                done = min(begin + len(chunk), len(todo))
                rate = done / max(time.time() - started, 1e-6)
                print(f"    text {done}/{len(todo)} {rate:.2f}/s", flush=True)

    todo = [index for index, sample in enumerate(samples) if sample.tweet_id not in done_image]
    if todo and "image" in want:
        started = time.time()
        with open(IMAGE_JSONL, "a", encoding="utf-8") as sink:
            for start in range(0, len(todo), args.batch):
                chunk = todo[start:start + args.batch]
                images = [Image.open(samples[index].image_path).convert("RGB") for index in chunk]
                tfeat, _ = encode_features(wrapper, [samples[index].text for index in chunk], images)
                pixels = wrapper.encode_image(images)["pixel_values"]
                values = image_occlusion(wrapper, pixels, tfeat)
                for row, index in enumerate(chunk):
                    sink.write(json.dumps({
                        "id": samples[index].tweet_id,
                        "method": "occlusion",
                        "values": [float(value) for value in values[row]],
                    }) + "\n")
                sink.flush()
                done = start + len(chunk)
                if done % 400 < args.batch or done == len(todo):
                    rate = done / max(time.time() - started, 1e-6)
                    print(f"    image {done}/{len(todo)} {rate:.2f}/s", flush=True)

    print("[2] done", flush=True)


if __name__ == "__main__":
    main()
