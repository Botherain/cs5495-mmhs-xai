"""P3.2-P3.3: TCAV on the image tower, with random-direction nulls and a
text-bias split.

CAV: logistic regression separating a concept's positive from negative images in
the image tower's 512-d pooler space. Sensitivity: the gradient of the analysis
target with respect to that pooler output, dotted with the unit CAV; TCAV is the
fraction of samples with a positive dot product.

Null: 20 random unit directions, so a concept only counts if it beats them.
Split-half: two CAVs trained on disjoint halves of the concept set, then compared
by the Spearman correlation of their per-sample sensitivities.
Text-bias control: TCAV recomputed separately on samples whose image contains
printed text and samples whose image does not.

Output: refine-logs/p3_tcav.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import encode_features, fusion_probs, spearman, target_of  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
N_NULL = 20


def load_concepts() -> dict:
    with open(os.path.join(OUT, "p3_concepts.json"), encoding="utf-8") as handle:
        return json.load(handle)


def load_text_presence() -> dict:
    presence = {}
    with open(os.path.join(OUT, "p0_ocr.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                presence[record["id"]] = bool(record.get("boxes"))
    return presence


def cav_direction(features: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    x = np.vstack([features[positives], features[negatives]])
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    model = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
    model.fit(x, y)
    direction = model.coef_[0]
    return direction / max(np.linalg.norm(direction), 1e-12)


def sensitivities(model, tfeat: torch.Tensor, vfeat: torch.Tensor, device: str,
                  batch: int = 128) -> np.ndarray:
    """d target / d vfeat for every sample, at that sample's own text feature."""
    gradients = []
    for start in range(0, vfeat.shape[0], batch):
        text = tfeat[start:start + batch].to(device)
        chunk = vfeat[start:start + batch].to(device).clone().requires_grad_(True)
        probs = fusion_probs(model, text, chunk)
        score = target_of(probs).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        gradients.append(chunk.grad.detach().cpu())
    return torch.cat(gradients).numpy()


def tcav_score(sensitivity: np.ndarray, direction: np.ndarray) -> float:
    return float((sensitivity @ direction > 0).mean())


def main() -> None:
    from PIL import Image

    concepts = load_concepts()["concepts"]
    presence = load_text_presence()
    with open(os.path.join(OUT, "p2_phi.jsonl"), encoding="utf-8") as handle:
        analysis_ids = [json.loads(line)["id"] for line in handle if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    analysis_images = [Image.open(os.path.join(ROOT, "img_resized", f"{t}.jpg")).convert("RGB")
                       for t in analysis_ids]
    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    analysis_texts = [gt[str(tweet_id)].get("tweet_text", "") for tweet_id in analysis_ids]
    analysis_tfeat, analysis_features = encode_features(wrapper, analysis_texts, analysis_images)
    print(f"[1] analysis={analysis_features.shape}", flush=True)

    sensitivity = sensitivities(model, analysis_tfeat, analysis_features, device)
    print(f"[2] sensitivities {sensitivity.shape}", flush=True)

    # How heterogeneous are the per-sample gradients? If they are nearly
    # collinear, "fraction of samples with a positive dot product" is a sign
    # test on one common direction and carries no per-sample information. The
    # mean cosine to the mean gradient is reported so the reader can judge the
    # TCAV number instead of trusting it.
    means = sensitivity.mean(axis=0)
    collinearity = float(np.mean(
        (sensitivity @ means) / (np.linalg.norm(sensitivity, axis=1) * np.linalg.norm(means) + 1e-12)
    ))
    print(f"    sensitivity collinearity (mean cosine to mean gradient): {collinearity:.4f}", flush=True)

    sensitivity = sensitivity / np.linalg.norm(sensitivity, axis=1, keepdims=True).clip(1e-12)

    rng = np.random.default_rng(0)
    null_directions = rng.normal(size=(N_NULL, sensitivity.shape[1]))
    null_directions /= np.linalg.norm(null_directions, axis=1, keepdims=True)
    null_scores = [tcav_score(sensitivity, direction) for direction in null_directions]
    null_mean = float(np.mean(null_scores))
    null_sd = float(np.std(null_scores, ddof=1))

    has_text = np.array([presence.get(tweet_id, False) for tweet_id in analysis_ids])

    result = {
        "n_analysis": len(analysis_ids),
        "n_with_text": int(has_text.sum()),
        "n_without_text": int((~has_text).sum()),
        "null": {"n": N_NULL, "mean": null_mean, "sd": null_sd, "scores": null_scores},
        "sensitivity_collinearity": collinearity,
        "concepts": {},
    }

    for name, block in concepts.items():
        concept_ids = list(block["positive"]) + list(block["negative"])
        images = [Image.open(os.path.join(ROOT, "img_resized", f"{t}.jpg")).convert("RGB")
                  for t in concept_ids]
        _, features = encode_features(wrapper, [""] * len(concept_ids), images)
        features = features.numpy()
        n_pos = len(block["positive"])
        positives = np.arange(n_pos)
        negatives = np.arange(n_pos, len(concept_ids))

        direction = cav_direction(features, positives, negatives)
        score = tcav_score(sensitivity, direction)
        z = (score - null_mean) / max(null_sd, 1e-9)
        p_value = float(np.mean(np.array(null_scores) >= score))

        # CAV validity: if the probe cannot separate the two concept halves it
        # cannot support a TCAV claim, whatever the score says.
        from sklearn.linear_model import LogisticRegression

        probe = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        probe.fit(features, np.concatenate([np.ones(n_pos), np.zeros(len(negatives))]))
        cav_accuracy = float(probe.score(features, np.concatenate([np.ones(n_pos), np.zeros(len(negatives))])))
        cosines = sensitivity @ direction

        # split-half: disjoint halves of the concept set
        half = n_pos // 2
        direction_a = cav_direction(features, positives[:half], negatives[:half])
        direction_b = cav_direction(features, positives[half:], negatives[half:])
        vector_a = sensitivity @ direction_a
        vector_b = sensitivity @ direction_b
        split_spearman = spearman(vector_a, vector_b)
        split_agreement = float(np.mean((vector_a > 0) == (vector_b > 0)))

        text_score = tcav_score(sensitivity[has_text], direction) if has_text.any() else None
        no_text_score = tcav_score(sensitivity[~has_text], direction) if (~has_text).any() else None

        result["concepts"][name] = {
            "n_positive": int(n_pos),
            "n_negative": int(len(negatives)),
            "tcav": score,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "z": float(z),
            "p_null": p_value,
            "split_half_spearman": split_spearman,
            "split_half_sign_agreement": split_agreement,
            "tcav_with_text": text_score,
            "tcav_without_text": no_text_score,
            "cav_accuracy": cav_accuracy,
            "mean_cosine": float(np.mean(cosines)),
            "cosine_sd": float(np.std(cosines)),
            "passes": bool(p_value < 0.05 and split_agreement > 0.5 and cav_accuracy >= 0.75),
            "cav_norm": float(np.linalg.norm(direction)),
        }
        print(f"    {name:13s} TCAV={score:.3f} null={null_mean:.3f}+-{null_sd:.3f} "
              f"cos={np.mean(cosines):+.3f}+-{np.std(cosines):.3f} cav_acc={cav_accuracy:.3f} "
              f"p={p_value:.3f} split={split_agreement:.2f} "
              f"text={text_score if text_score is None else round(text_score, 3)} "
              f"no-text={no_text_score if no_text_score is None else round(no_text_score, 3)}", flush=True)

    with open(os.path.join(OUT, "p3_tcav.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("[3] wrote p3_tcav.json")


if __name__ == "__main__":
    main()
