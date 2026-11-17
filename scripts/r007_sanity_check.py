"""R007: parameter-randomisation sanity check for occlusion attributions.

Progressively re-initialises the top-k transformer layers of each tower and
recomputes occlusion importances. A faithful explanation changes when the
parameters change, so the Spearman correlation with the original attribution
should fall as more layers are randomised.

Usage: python scripts/r007_sanity_check.py [--n 100] [--restart]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.faithfulness import content_positions, patch_grid  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
CHECKPOINT = os.path.join(OUT, "r007_sanity_check.jsonl")
FINAL = os.path.join(OUT, "r007_sanity_check.json")
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "r007_sanity_check.log")
LEVELS = (0, 3, 6, 9, 12)


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def randomise_layers(tower, from_layer: int) -> None:
    """Re-initialise transformer layers from_layer..end (top-down cascade)."""
    layers = tower.encoder.layers
    for index in range(from_layer, len(layers)):
        for module in layers[index].modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)


def text_importance(wrapper, ids, mask, pix, positions):
    """Leave-one-out importance; hiding a token flips its mask bit, ids unchanged."""
    base = wrapper.target(ids, mask, pix).item()
    values = []
    for position in positions:
        probe_mask = mask.clone()
        probe_mask[0, position] = 0
        values.append(base - wrapper.target(ids, probe_mask, pix).item())
    return np.asarray(values)


def image_importance(wrapper, ids, mask, pix):
    base = wrapper.target(ids, mask, pix).item()
    values = []
    for row_start, row_end, col_start, col_end in patch_grid(pix.shape[-1], 7):
        probe = pix.clone()
        probe[0, :, row_start:row_end, col_start:col_end] = 0.0
        values.append(base - wrapper.target(ids, mask, probe).item())
    return np.asarray(values)


def read_checkpoint() -> dict:
    records = {}
    if not os.path.exists(CHECKPOINT):
        return records
    with open(CHECKPOINT, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[record["tweet_id"]] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    from PIL import Image
    from scipy import stats

    os.makedirs(OUT, exist_ok=True)
    if args.restart and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(os.environ["MMHS_MODEL_DIR"], device=device)
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["selection"][:args.n]
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)

    samples = []
    for tweet_id in ids:
        image_path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        if not os.path.exists(image_path):
            continue
        entry = gt[str(tweet_id)]
        samples.append(Sample(
            tweet_id=str(tweet_id), text=(entry.get("tweet_text") or "").strip(),
            ocr_text="", image_path=image_path,
            labels=normalize_labels(entry.get("labels")),
        ))

    done = read_checkpoint()
    pending = [s for s in samples if s.tweet_id not in done]
    log(f"start: {len(samples)} samples, {len(done)} done, {len(pending)} pending")

    text_state = copy.deepcopy(model.tower_txt.state_dict())
    image_state = copy.deepcopy(model.tower_img.state_dict())
    started = time.time()

    for index, sample in enumerate(pending):
        encoded = wrapper.encode_text([sample.text or ""])
        ids_t, mask_t = encoded["input_ids"], encoded["attention_mask"]
        image = Image.open(sample.image_path).convert("RGB")
        pix = wrapper.encode_image([image])["pixel_values"]
        positions = content_positions(mask_t, tokenizer)

        base_text = text_importance(wrapper, ids_t, mask_t, pix, positions)
        base_image = image_importance(wrapper, ids_t, mask_t, pix)

        record = {"tweet_id": sample.tweet_id, "n_tokens": len(positions), "levels": {}}
        for level in LEVELS:
            # text tower randomised alone -> text attribution curve
            torch.nn.Module.load_state_dict(model.tower_txt, text_state)
            torch.nn.Module.load_state_dict(model.tower_img, image_state)
            randomise_layers(model.tower_txt, level)
            perturbed_text = text_importance(wrapper, ids_t, mask_t, pix, positions)

            # image tower randomised alone -> image attribution curve
            torch.nn.Module.load_state_dict(model.tower_txt, text_state)
            torch.nn.Module.load_state_dict(model.tower_img, image_state)
            randomise_layers(model.tower_img, level)
            perturbed_image = image_importance(wrapper, ids_t, mask_t, pix)

            rho_text = stats.spearmanr(base_text, perturbed_text).statistic if len(base_text) > 2 else float("nan")
            rho_image = stats.spearmanr(base_image, perturbed_image).statistic

            record["levels"][str(level)] = {
                "spearman_text": float(rho_text),
                "spearman_image": float(rho_image),
                "max_abs_delta_text": float(np.abs(perturbed_text - base_text).max()),
                "max_abs_delta_image": float(np.abs(perturbed_image - base_image).max()),
            }

        torch.nn.Module.load_state_dict(model.tower_txt, text_state)
        torch.nn.Module.load_state_dict(model.tower_img, image_state)

        with open(CHECKPOINT, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        if (index + 1) % 10 == 0:
            rate = (index + 1) / max(time.time() - started, 1e-6)
            log(f"{index + 1}/{len(pending)} done, {rate:.2f}/s, eta {(len(pending) - index - 1) / rate / 60:.1f} min")

    records = list(read_checkpoint().values())
    if not records:
        log("no records")
        return
    summary = {"n": len(records), "levels": {}}
    for level in LEVELS:
        text_rho = [r["levels"][str(level)]["spearman_text"] for r in records]
        image_rho = [r["levels"][str(level)]["spearman_image"] for r in records]
        summary["levels"][str(level)] = {
            "mean_spearman_text": float(np.nanmean(text_rho)),
            "mean_spearman_image": float(np.nanmean(image_rho)),
        }
    with open(FINAL, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2, ensure_ascii=False)
    for level in LEVELS:
        values = summary["levels"][str(level)]
        log(f"level {level:2d}: text rho={values['mean_spearman_text']:.3f} image rho={values['mean_spearman_image']:.3f}")


if __name__ == "__main__":
    main()
