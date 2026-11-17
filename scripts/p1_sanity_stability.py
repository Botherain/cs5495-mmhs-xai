"""P1.4-P1.6: gates (parameter randomisation), stability, inter-method agreement.

Gates come first, as pre-registered: an attribution that survives a random model
is not explaining this model. Each tower is randomised on its own --- 0, 3, 6, 9
or all 12 encoder blocks --- and the attribution is compared with the trained
model's by Spearman correlation over units.

Stability: LIME / KernelSHAP under three seeds, and all four image methods under
5% Gaussian pixel noise. Agreement: pairwise Spearman and top-10% overlap on the
full selection set, from the stored attributions.

Outputs
  refine-logs/p1_sanity_curves.json
  refine-logs/p1_stability.json
  refine-logs/p1_agreement.json

Usage: python scripts/p1_sanity_stability.py [--n 100]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import (  # noqa: E402
    IMAGE_METHODS, TEXT_METHODS, content_positions, encode_features, image_gradcam,
    image_ig, image_occlusion, image_scorecam, spearman, text_kernelshap, text_lime,
    text_occlusion, topk_overlap,
)
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
LEVELS = (0, 3, 6, 9, 12)


def load_samples(limit: int = 0) -> list:
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        ids = json.load(handle)["selection"]
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


def load_jsonl(path: str) -> dict:
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["id"]] = record
    return records


def randomise_tower(model, tower_name: str, n_layers: int) -> None:
    """Re-initialise the last ``n_layers`` encoder blocks of one tower."""
    if n_layers == 0:
        return
    tower = getattr(model, tower_name)
    layers = tower.encoder.layers
    for layer in layers[len(layers) - n_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(0.0, 0.02)


def snapshot(model, tower_name: str) -> dict:
    return {key: value.detach().clone() for key, value in getattr(model, tower_name).state_dict().items()}


def restore(model, tower_name: str, state: dict) -> None:
    getattr(model, tower_name).load_state_dict(state)


def text_attributions(wrapper, ids_tensor, mask_tensor, tfeat, vfeat, indices, seeds=(0,)):
    """method -> (n_samples, n_tokens) for one seed."""
    results = {name: [] for name in TEXT_METHODS}
    for index in indices:
        ids_row, mask_row = ids_tensor[index], mask_tensor[index]
        positions = content_positions(mask_row)
        for name in TEXT_METHODS:
            if name == "occlusion":
                values = text_occlusion(wrapper, ids_row, mask_row, positions, vfeat[index])
            elif name == "lime":
                values = text_lime(wrapper, ids_row, mask_row, positions, vfeat[index], seed=seeds[0])
            else:
                values = text_kernelshap(wrapper, ids_row, mask_row, positions, vfeat[index], seed=seeds[0])
            results[name].append(np.asarray(values, dtype=np.float64))
    return results


def image_attributions(wrapper, pixels, tfeat, indices):
    results = {name: [] for name in IMAGE_METHODS}
    for index in indices:
        pixel, text = pixels[index:index + 1], tfeat[index]
        results["occlusion"].append(image_occlusion(wrapper, pixel, text)[0])
        results["gradcam"].append(image_gradcam(wrapper, pixel, text)[0])
        results["ig"].append(image_ig(wrapper, pixel, text)[0])
        results["scorecam"].append(image_scorecam(wrapper, pixel, text)[0])
    return results


def compare(reference, perturbed) -> tuple:
    correlations, overlaps = [], []
    for base, other in zip(reference, perturbed):
        if len(base) != len(other) or len(base) < 3:
            continue
        correlations.append(spearman(base, other))
        overlaps.append(topk_overlap(base, other, 0.1))
    values = [value for value in correlations if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan"), float(np.mean(overlaps)) if overlaps else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="samples for gates and stability")
    parser.add_argument("--seeds", type=int, default=3, help="repeats for the randomised methods")
    parser.add_argument("--noise", type=int, default=3, help="noise draws for the image methods")
    parser.add_argument("--modalities", default="text,image", help="which gate curves to build")
    parser.add_argument("--stage", default="gates,stability,agreement",
                        help="any of gates, stability, agreement")
    args = parser.parse_args()
    want = [item.strip() for item in args.modalities.split(",") if item.strip()]
    stages = {item.strip() for item in args.stage.split(",") if item.strip()}

    from PIL import Image

    samples = load_samples(args.n)
    indices = list(range(len(samples)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    os.makedirs(OUT, exist_ok=True)
    started = time.time()

# gate curves, resumable
    # The process has died silently twice on long runs here, so every completed
    # randomisation level is flushed to disk and skipped on the next attempt.
    curves_path = os.path.join(OUT, "p1_sanity_curves.json")
    saved = {}
    if os.path.exists(curves_path) and "gates" in stages:
        with open(curves_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("n") != len(samples):
            print(f"[0] ignoring {curves_path}: n={saved.get('n')} != {len(samples)}", flush=True)
            saved = {}
    curves = {"text": {}, "image": {}}
    for modality in ("text", "image"):
        for level, block in (saved.get("curves", {}).get(modality) or {}).items():
            curves[modality][int(level)] = block

    def dump_gates(gates=None) -> None:
        payload = {"n": len(samples), "levels": list(LEVELS), "curves": curves}
        if gates is not None:
            payload["gates"] = gates
        elif saved.get("gates"):
            payload["gates"] = saved["gates"]
        with open(curves_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print(f"[1] samples={len(samples)} want={want} stages={sorted(stages)} "
          f"done_text={sorted(curves['text'])} done_image={sorted(curves['image'])}", flush=True)

    images = [Image.open(sample.image_path).convert("RGB") for sample in samples]
    encoded_text = wrapper.encode_text([sample.text for sample in samples])
    ids_tensor, mask_tensor = encoded_text["input_ids"], encoded_text["attention_mask"]
    pixels = wrapper.encode_image(images)["pixel_values"]
    tfeat, vfeat = encode_features(wrapper, [s.text for s in samples], images)

    base = {}
    need_text_base = ("text" in want and "gates" in stages) or ("stability" in stages)
    need_image_base = ("image" in want and "gates" in stages) or ("stability" in stages)

    if need_text_base:
        state = snapshot(model, "tower_txt")
        base["text"] = text_attributions(wrapper, ids_tensor, mask_tensor, tfeat, vfeat, indices)
        if "text" in want and "gates" in stages:
            for level in LEVELS:
                if level in curves["text"]:
                    continue
                if level:
                    torch.manual_seed(level)
                    randomise_tower(model, "tower_txt",
                                    level if level < 12 else len(model.tower_txt.encoder.layers))
                perturbed = text_attributions(wrapper, ids_tensor, mask_tensor, tfeat, vfeat, indices)
                curves["text"][level] = {}
                for name in TEXT_METHODS:
                    rho, overlap = compare(base["text"][name], perturbed[name])
                    curves["text"][level][name] = {"spearman": rho, "top10_overlap": overlap}
                restore(model, "tower_txt", state)
                dump_gates()
                print(f"    text randomise {level:2d} blocks -> "
                      + " ".join(f"{n}:{curves['text'][level][n]['spearman']:.2f}" for n in TEXT_METHODS)
                      + f" ({time.time() - started:.0f}s)", flush=True)
            restore(model, "tower_txt", state)

    if need_image_base:
        state = snapshot(model, "tower_img")
        base["image"] = image_attributions(wrapper, pixels, tfeat, indices)
        if "image" in want and "gates" in stages:
            for level in LEVELS:
                if level in curves["image"]:
                    continue
                if level:
                    torch.manual_seed(100 + level)
                    randomise_tower(model, "tower_img",
                                    level if level < 12 else len(model.tower_img.encoder.layers))
                perturbed = image_attributions(wrapper, pixels, tfeat, indices)
                curves["image"][level] = {}
                for name in IMAGE_METHODS:
                    rho, overlap = compare(base["image"][name], perturbed[name])
                    curves["image"][level][name] = {"spearman": rho, "top10_overlap": overlap}
                restore(model, "tower_img", state)
                dump_gates()
                print(f"    image randomise {level:2d} blocks -> "
                      + " ".join(f"{n}:{curves['image'][level][n]['spearman']:.2f}" for n in IMAGE_METHODS)
                      + f" ({time.time() - started:.0f}s)", flush=True)
            restore(model, "tower_img", state)

    if "gates" in stages:
        from scipy.stats import spearmanr

        complete = all(len(curves[m]) == len(LEVELS) for m in ("text", "image"))
        if complete:
            gates = {"text": {}, "image": {}}
            for modality, methods in (("text", TEXT_METHODS), ("image", IMAGE_METHODS)):
                for name in methods:
                    values = [curves[modality][level][name]["spearman"] for level in LEVELS]
                    rho_depth = spearmanr(LEVELS, values).statistic
                    full = values[-1]
                    gates[modality][name] = {
                        "rho_depth": float(rho_depth) if np.isfinite(rho_depth) else None,
                        "r_full": float(full) if np.isfinite(full) else None,
                        "passes": bool(np.isfinite(rho_depth) and rho_depth < -0.7
                                       and np.isfinite(full) and full < 0.2),
                    }
                    print(f"    gate {modality}/{name}: rho_depth={rho_depth:+.2f} "
                          f"r_full={full:+.2f}", flush=True)
            dump_gates(gates)
        else:
            dump_gates()
            print(f"[2] gate curves incomplete: text={sorted(curves['text'])} "
                  f"image={sorted(curves['image'])}", flush=True)

# stability
    # Pre-registered: every randomised method is repeated, every method is
    # compared against the clean baseline under 5% input noise, and the reported
    # number is the *worst* pair, not the average.
    if "stability" in stages:
        stability_path = os.path.join(OUT, "p1_stability.json")
        stability = {"text_seeds": {}, "image_noise": {},
                     "criterion": "rho>0.5 and top10_overlap>=0.5 (worst pair)"}
        if os.path.exists(stability_path):
            with open(stability_path, encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous.get("n") == len(samples):
                stability["text_seeds"] = previous.get("text_seeds", {})
                stability["image_noise"] = previous.get("image_noise", {})

        def dump_stability() -> None:
            with open(stability_path, "w", encoding="utf-8") as handle:
                json.dump({"n": len(samples), **stability}, handle, indent=2)

        for name in ("lime", "kernelshap"):
            block = stability["text_seeds"].get(name)
            if block and len(block.get("pairs", [])) >= args.seeds:
                continue
            pairs = []
            for seed in range(1, args.seeds + 1):
                rerun = text_attributions(wrapper, ids_tensor, mask_tensor, tfeat, vfeat,
                                          indices, seeds=(seed,))[name]
                rho, overlap = compare(base["text"][name], rerun)
                pairs.append({"seed": seed, "spearman": rho, "top10_overlap": overlap})
                stability["text_seeds"][name] = {
                    "pairs": pairs,
                    "min_spearman": min(pair["spearman"] for pair in pairs),
                    "min_top10_overlap": min(pair["top10_overlap"] for pair in pairs),
                    "passes": bool(min(pair["spearman"] for pair in pairs) > 0.5
                                   and min(pair["top10_overlap"] for pair in pairs) >= 0.5),
                }
                dump_stability()
                print(f"    seed stability {name} seed={seed}: rho={rho:.3f} "
                      f"overlap={overlap:.3f}", flush=True)

        for name in IMAGE_METHODS:
            block = stability["image_noise"].get(name)
            if block and len(block.get("pairs", [])) >= args.noise:
                continue
            pairs = []
            for repeat in range(args.noise):
                generator = torch.Generator().manual_seed(7 + repeat)
                noisy = (pixels + 0.05 * torch.randn(pixels.shape, generator=generator)).clamp(-3, 3)
                noisy_result = image_attributions(wrapper, noisy, tfeat, indices)
                rho, overlap = compare(base["image"][name], noisy_result[name])
                pairs.append({"repeat": repeat, "spearman": rho, "top10_overlap": overlap})
                stability["image_noise"][name] = {
                    "pairs": pairs,
                    "min_spearman": min(pair["spearman"] for pair in pairs),
                    "min_top10_overlap": min(pair["top10_overlap"] for pair in pairs),
                    "passes": bool(min(pair["spearman"] for pair in pairs) > 0.5
                                   and min(pair["top10_overlap"] for pair in pairs) >= 0.5),
                }
                dump_stability()
                print(f"    noise stability {name} repeat={repeat}: rho={rho:.3f} "
                      f"overlap={overlap:.3f}", flush=True)

# inter-method agreement (full selection set)
    if "agreement" in stages:
        text_records = load_jsonl(os.path.join(OUT, "p1_text_attr.jsonl"))
        image_records = load_jsonl(os.path.join(OUT, "p1_image_attr.jsonl"))
        agreement = {"text": {}, "image": {}}
        for modality, records, methods in (
            ("text", text_records, TEXT_METHODS),
            ("image", image_records, IMAGE_METHODS),
        ):
            available = [record for record in records.values()]
            for i, first in enumerate(methods):
                for second in methods[i + 1:]:
                    correlations, overlaps = [], []
                    for record in available:
                        if modality == "text":
                            first_values = record[first][:record["n_tokens"]]
                            second_values = record[second][:record["n_tokens"]]
                        else:
                            first_values = record[first]
                            second_values = record[second]
                        correlations.append(spearman(first_values, second_values))
                        overlaps.append(topk_overlap(first_values, second_values, 0.1))
                    values = [value for value in correlations if np.isfinite(value)]
                    agreement[modality][f"{first}__{second}"] = {
                        "spearman": float(np.mean(values)) if values else float("nan"),
                        "top10_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
                        "n": len(overlaps),
                    }
            print(f"    agreement {modality}: " + json.dumps(agreement[modality]), flush=True)

        with open(os.path.join(OUT, "p1_agreement.json"), "w", encoding="utf-8") as handle:
            json.dump(agreement, handle, indent=2)

    print(f"[3] done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
