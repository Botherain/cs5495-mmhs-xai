"""P2: modality-level coalition attribution on the 3,000-sample analysis set.

Two players, four coalitions, exact Shapley from four forward passes:

    phi_T = 1/2 [v(T) - v(empty)] + 1/2 [v(T,I) - v(I)]
    phi_I = 1/2 [v(I) - v(empty)] + 1/2 [v(T,I) - v(T)]
    I     = v(T,I) - v(T) - v(I) + v(empty)

Blank content, never absence: v(empty) is the empty string's text feature and the
mean-colour image, both pushed through their own towers. The alternative A1
reading uses the checkpoint's own presence flags (text_present / image_present),
which switch the fusion to a single tower --- that is a different counterfactual
and can disagree; both are stored.

Modality deletion curves interpolate each tower's feature towards its blank value
in five steps, with five random directions of equal norm as the null.

Outputs
  refine-logs/p2_phi.jsonl       per sample: coalitions, phi, interaction, gate
  refine-logs/p2_curves.json     deletion curves + random-direction null
  refine-logs/p2_summary.json    quantiles, shares, tests

Usage: python scripts/p2_modality.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import encode_features, fusion_probs, target_of  # noqa: E402
from mmhs.data import Sample, normalize_labels  # noqa: E402
from mmhs.model import TASK_NAMES, Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p2_phi.jsonl")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RANDOM_DIRECTIONS = 5


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


def agreement(sample: Sample) -> float:
    labels = sample.labels
    unique = len(set(labels))
    if unique == 1:
        return 1.0
    if unique == len(labels):
        return 1.0 / 3.0
    return 2.0 / 3.0


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
    args = parser.parse_args()

    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    samples = load_samples(args.limit)
    finished = done_ids()
    todo = [index for index, sample in enumerate(samples) if sample.tweet_id not in finished]
    print(f"[1] samples={len(samples)} done={len(finished)} todo={len(todo)}", flush=True)
    if not todo:
        return

    images = [Image.open(sample.image_path).convert("RGB") for sample in samples]
    tfeat, vfeat = encode_features(wrapper, [s.text for s in samples], images)

    with torch.no_grad():
        blank_ids, blank_mask = wrapper.blank_text(1)
        tfeat_blank = model._encode_text(blank_ids.to(device), blank_mask.to(device)).cpu().squeeze(0)
        blank_pixel = torch.zeros(1, 3, 224, 224, device=device)
        vfeat_blank = model._encode_image(blank_pixel).cpu().squeeze(0)
    print(f"[2] blank features ready (tfeat_blank norm {tfeat_blank.norm():.3f}, "
          f"vfeat_blank norm {vfeat_blank.norm():.3f})", flush=True)

    os.makedirs(OUT, exist_ok=True)
    curves = {"ids": [], "text": [], "image": [], "random": []}
    with open(JSONL, "a", encoding="utf-8") as sink, torch.no_grad():
        for step, index in enumerate(todo, start=1):
            sample = samples[index]
            text_feature = tfeat[index].to(device)
            image_feature = vfeat[index].to(device)
            text_blank = tfeat_blank.to(device)
            image_blank = vfeat_blank.to(device)

            def value(t: torch.Tensor, v: torch.Tensor, flags=None) -> torch.Tensor:
                tf = t.unsqueeze(0)
                vf = v.unsqueeze(0)
                if flags is None:
                    return fusion_probs(model, tf, vf)[0]
                text_present, image_present = flags
                return fusion_probs(model, tf, vf,
                                    text_present=torch.tensor([text_present], device=device),
                                    image_present=torch.tensor([image_present], device=device))[0]

            v_empty = value(text_blank, image_blank)
            v_text = value(text_feature, image_blank)
            v_image = value(text_blank, image_feature)
            v_both = value(text_feature, image_feature)

            phi_t = 0.5 * (v_text - v_empty) + 0.5 * (v_both - v_image)
            phi_i = 0.5 * (v_image - v_empty) + 0.5 * (v_both - v_text)
            interaction = v_both - v_text - v_image + v_empty

            # A1: the checkpoint's own missing-modality path
            f_empty = value(text_blank, image_blank, (0.0, 0.0))
            f_text = value(text_feature, image_blank, (1.0, 0.0))
            f_image = value(text_blank, image_feature, (0.0, 1.0))
            f_both = value(text_feature, image_feature, (1.0, 1.0))
            phi_t_flag = 0.5 * (f_text - f_empty) + 0.5 * (f_both - f_image)
            phi_i_flag = 0.5 * (f_image - f_empty) + 0.5 * (f_both - f_text)

            encoded_text = wrapper.encode_text([sample.text])
            text_norm, image_norm = wrapper.gate_stats(
                encoded_text["input_ids"],
                encoded_text["attention_mask"],
                wrapper.encode_image([images[index]])["pixel_values"],
            )

            record = {
                "id": sample.tweet_id,
                "v_empty": float(target_of(v_empty)),
                "v_text": float(target_of(v_text)),
                "v_image": float(target_of(v_image)),
                "v_both": float(target_of(v_both)),
                "phi_t": float(target_of(phi_t)),
                "phi_i": float(target_of(phi_i)),
                "interaction": float(target_of(interaction)),
                "phi_t_flag": float(target_of(phi_t_flag)),
                "phi_i_flag": float(target_of(phi_i_flag)),
                "v_empty_flag": float(target_of(f_empty)),
                "v_both_flag": float(target_of(f_both)),
                "target": float(target_of(v_both)),
                "heads_both": [float(x) for x in v_both],
                "heads_empty": [float(x) for x in v_empty],
                "gate_text_norm": float(text_norm),
                "gate_image_norm": float(image_norm),
                "agree": agreement(sample),
                "label_heads": sorted({int(label - 1) for label in sample.labels if label > 0}),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()

            # deletion curves: interpolate one feature towards its blank value
            curves["ids"].append(sample.tweet_id)
            direction_text = (text_feature - text_blank)
            direction_image = (image_feature - image_blank)
            curve_text, curve_image = [], []
            for alpha in ALPHAS:
                curve_text.append(float(target_of(value(text_blank + alpha * direction_text, image_feature))))
                curve_image.append(float(target_of(value(text_feature, image_blank + alpha * direction_image))))
            curves["text"].append(curve_text)
            curves["image"].append(curve_image)

            rng = np.random.default_rng(index)
            random_curves = []
            for _ in range(RANDOM_DIRECTIONS):
                unit_t = torch.from_numpy(rng.normal(size=text_feature.shape).astype(np.float32)).to(device)
                unit_t = unit_t / unit_t.norm() * direction_text.norm()
                unit_v = torch.from_numpy(rng.normal(size=image_feature.shape).astype(np.float32)).to(device)
                unit_v = unit_v / unit_v.norm() * direction_image.norm()
                random_curves.append([
                    float(target_of(value(text_feature + alpha * unit_t, image_feature + alpha * unit_v)))
                    for alpha in ALPHAS
                ])
            curves["random"].append(random_curves)

            if step % 200 == 0 or step == len(todo):
                print(f"    {step}/{len(todo)}", flush=True)

    with open(os.path.join(OUT, "p2_curves.json"), "w", encoding="utf-8") as handle:
        json.dump({"alphas": list(ALPHAS), **curves}, handle)

    summarise()


def summarise() -> None:
    from scipy.stats import spearmanr, wilcoxon

    rows = []
    with open(JSONL, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def array(key):
        return np.array([row[key] for row in rows], dtype=np.float64)

    phi_t, phi_i = array("phi_t"), array("phi_i")
    interaction = array("interaction")
    dominated = int((np.abs(phi_t) > np.abs(phi_i)).sum())
    positive_phi_i = int((phi_i > 0).sum())
    positive_interaction = int((interaction > 0).sum())

    gate_text, gate_image = array("gate_text_norm"), array("gate_image_norm")
    ratio_text = np.abs(phi_t) / (np.abs(phi_t) + np.abs(phi_i) + 1e-12)
    gate_share = gate_text / (gate_text + gate_image + 1e-12)
    gate_rho = spearmanr(gate_share, ratio_text).statistic

    completeness = np.abs(phi_t + phi_i - (array("v_both") - array("v_empty")))
    flag_agree = float(np.mean(np.sign(phi_t) == np.sign(array("phi_t_flag"))))

    curves = json.load(open(os.path.join(OUT, "p2_curves.json"), encoding="utf-8"))
    text_curves = np.array(curves["text"], dtype=np.float64)
    image_curves = np.array(curves["image"], dtype=np.float64)
    random_curves = np.array(curves["random"], dtype=np.float64)

    def area(values):
        return float(np.mean(values - values[..., :1], axis=-1).mean())

    text_drop = float(np.mean(text_curves[:, -1] - text_curves[:, 0]))
    image_drop = float(np.mean(image_curves[:, -1] - image_curves[:, 0]))
    random_drop = float(np.mean(random_curves[:, :, 0] - random_curves[:, :, -1]))

    def paired_p(values, reference):
        diff = values - reference
        diff = diff[np.isfinite(diff)]
        if diff.size < 10 or not np.any(diff != 0):
            return None
        return float(wilcoxon(diff).pvalue)

    text_reference = (random_curves[:, :, 0] - random_curves[:, :, -1]).mean(axis=1)
    image_reference = text_reference
    summary = {
        "n": len(rows),
        "phi_t": {
            "mean": float(phi_t.mean()), "median": float(np.median(phi_t)),
            "q05": float(np.quantile(phi_t, 0.05)), "q95": float(np.quantile(phi_t, 0.95)),
            "share_positive": float((phi_t > 0).mean()),
        },
        "phi_i": {
            "mean": float(phi_i.mean()), "median": float(np.median(phi_i)),
            "q05": float(np.quantile(phi_i, 0.05)), "q95": float(np.quantile(phi_i, 0.95)),
            "share_positive": float((phi_i > 0).mean()),
        },
        "text_dominance_share": dominated / len(rows),
        "phi_i_negative_share": 1.0 - positive_phi_i / len(rows),
        "interaction_positive_share": positive_interaction / len(rows),
        "interaction_mean": float(interaction.mean()),
        "interaction_median": float(np.median(interaction)),
        "completeness_max_error": float(completeness.max()),
        "gate_spearman": float(gate_rho),
        "gate_spearman_p": float(spearmanr(gate_share, ratio_text).pvalue),
        "flag_phi_t_sign_agreement": flag_agree,
        "deletion": {
            "text_reduction": text_drop,
            "image_reduction": image_drop,
            "random_reduction": random_drop,
            "text_minus_image": text_drop - image_drop,
            "text_vs_random_p": paired_p(text_curves[:, -1] - text_curves[:, 0], text_reference),
            "image_vs_random_p": paired_p(image_curves[:, -1] - image_curves[:, 0], image_reference),
        },
        "heads_empty_mean": np.mean([r["heads_empty"] for r in rows], axis=0).tolist(),
        "heads_both_mean": np.mean([r["heads_both"] for r in rows], axis=0).tolist(),
        "head_names": TASK_NAMES,
    }
    with open(os.path.join(OUT, "p2_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[3] summary")
    print(f"    phi_T mean {summary['phi_t']['mean']:+.4f} median {summary['phi_t']['median']:+.4f}")
    print(f"    phi_I mean {summary['phi_i']['mean']:+.4f} median {summary['phi_i']['median']:+.4f}")
    print(f"    text-dominant {summary['text_dominance_share']:.3%}  "
          f"phi_I<0 {summary['phi_i_negative_share']:.1%}  "
          f"interaction>0 {summary['interaction_positive_share']:.1%}")
    print(f"    completeness max error {summary['completeness_max_error']:.2e}")
    print(f"    gate rho {summary['gate_spearman']:+.3f} (p {summary['gate_spearman_p']:.2e})")
    print(f"    deletion text {text_drop:+.4f} image {image_drop:+.4f} random {random_drop:+.4f}")


if __name__ == "__main__":
    main()
