"""P1.1: text-side attributions on the 499-sample selection set.

Three candidates, one unit (content BPE token), one deletion operator (attention
mask flip):

  occlusion   leave-one-token-out
  lime        LIME over token-presence masks, text kernel (width 25)
  kernelshap  sampled coalitions, SHAP kernel, paired complements

Attributions are stored on the 77-slot row so later stages can rebuild the exact
input; only content positions are filled.

Output: refine-logs/p1_text_attr.jsonl (one line per sample, resumable)
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
    TEXT_METHODS, content_positions, encode_features, text_kernelshap,
    text_lime, text_occlusion,
)
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p1_text_attr.jsonl")


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
    parser.add_argument("--lime-samples", type=int, default=1500)
    parser.add_argument("--shap-samples", type=int, default=2048)
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

    tfeat, vfeat = encode_features(wrapper, [s.text for s in samples], images)
    encoded = wrapper.encode_text([s.text for s in samples])
    ids_tensor = encoded["input_ids"]
    mask_tensor = encoded["attention_mask"]

    os.makedirs(OUT, exist_ok=True)
    started = time.time()
    with open(JSONL, "a", encoding="utf-8") as sink:
        for step, index in enumerate(todo, start=1):
            sample = samples[index]
            ids_row = ids_tensor[index]
            mask_row = mask_tensor[index]
            positions = content_positions(mask_row)
            fixed_v = vfeat[index]
            record = {"id": sample.tweet_id, "n_tokens": len(positions), "positions": positions}
            for name in TEXT_METHODS:
                if name == "occlusion":
                    values = text_occlusion(wrapper, ids_row, mask_row, positions, fixed_v)
                elif name == "lime":
                    values = text_lime(wrapper, ids_row, mask_row, positions, fixed_v,
                                       num_samples=args.lime_samples)
                else:
                    values = text_kernelshap(wrapper, ids_row, mask_row, positions, fixed_v,
                                             num_samples=args.shap_samples)
                record[name] = [float(value) for value in values]
            sink.write(json.dumps(record) + "\n")
            sink.flush()

            if step % 25 == 0 or step == len(todo):
                rate = step / max(time.time() - started, 1e-6)
                print(
                    f"    {step}/{len(todo)} {rate:.2f}/s eta {(len(todo) - step) / rate / 60:.1f} min",
                    flush=True,
                )

    print("[2] done", flush=True)


if __name__ == "__main__":
    main()
