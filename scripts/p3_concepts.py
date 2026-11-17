"""P3.0: concept sets for TCAV, curated from image content (no model involved).

Three concepts, each with positive and negative examples drawn from the 3,499
analysed images:

  printed_text  positive: EasyOCR keeps at least one box at confidence >= 0.6;
                negative: the detector returned no box at any confidence.
  face          positive: the OpenCV Haar frontal-face cascade fires; negative: no
                face at any scale.
  flat_graphic  positive: the image sits in the bottom decile of normalised colour
                entropy (memes, screenshots, flat backgrounds); negative: top
                decile (photographic scenes).

Every cue is independent of the audited model, so the CAV direction is not
circular. A 20% sample of each set is written out for the hand check.

Output: refine-logs/p3_concepts.json
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
PER_CONCEPT = 400


def colour_entropy(path: str) -> float:
    from PIL import Image

    with Image.open(path) as image:
        small = image.convert("RGB").resize((96, 96))
        array = np.asarray(small, dtype=np.uint8)
    quantised = (array >> 4).astype(np.int32)
    codes = quantised[..., 0] * 256 + quantised[..., 1] * 16 + quantised[..., 2]
    counts = np.bincount(codes.ravel(), minlength=4096).astype(np.float64)
    share = counts[counts > 0] / counts.sum()
    return float(-(share * np.log2(share)).sum() / 12.0)


def main() -> None:
    import cv2

    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        subset = json.load(handle)
    ids = list(dict.fromkeys(subset["main"] + subset["selection"]))

    ocr = {}
    with open(os.path.join(OUT, "p0_ocr.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                ocr[record["id"]] = record

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print(f"[1] ids={len(ids)} ocr records={len(ocr)}", flush=True)

    text_pos, text_neg = [], []
    face_pos, face_neg = [], []
    entropy = {}
    entropy_failed = 0
    for index, tweet_id in enumerate(ids, start=1):
        path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
        record = ocr.get(tweet_id, {})
        if record.get("boxes"):
            text_pos.append(tweet_id)
        elif record.get("n_raw", 1) == 0:
            text_neg.append(tweet_id)

        frame = cv2.imread(path)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces):
                face_pos.append(tweet_id)
            else:
                face_neg.append(tweet_id)

        try:
            entropy[tweet_id] = colour_entropy(path)
        except Exception:
            entropy_failed += 1
        if index % 500 == 0:
            print(f"    {index}/{len(ids)} text+={len(text_pos)} face+={len(face_pos)}", flush=True)

    if entropy_failed:
        print(f"    entropy failed on {entropy_failed} images", flush=True)
    ranked = sorted(entropy.items(), key=lambda item: item[1])
    decile = max(1, len(ranked) // 10)
    flat_pos = [tweet_id for tweet_id, _ in ranked[:decile]]
    flat_neg = [tweet_id for tweet_id, _ in ranked[-decile:]]

    rng = np.random.default_rng(0)

    def pick(pool):
        pool = list(pool)
        if len(pool) <= PER_CONCEPT:
            return pool
        chosen = rng.choice(len(pool), size=PER_CONCEPT, replace=False)
        return sorted(pool[index] for index in chosen)

    concepts = {
        "printed_text": {"positive": pick(text_pos), "negative": pick(text_neg),
                         "pool": {"positive": len(text_pos), "negative": len(text_neg)},
                         "cue": "EasyOCR box at confidence >= 0.6 vs no detection at all"},
        "face": {"positive": pick(face_pos), "negative": pick(face_neg),
                 "pool": {"positive": len(face_pos), "negative": len(face_neg)},
                 "cue": "OpenCV Haar frontal-face cascade"},
        "flat_graphic": {"positive": pick(flat_pos), "negative": pick(flat_neg),
                         "pool": {"positive": len(flat_pos), "negative": len(flat_neg)},
                         "cue": "bottom vs top decile of normalised colour entropy"},
    }
    for name, block in concepts.items():
        print(f"[2] {name:12s} pos={len(block['positive'])} neg={len(block['negative'])} "
              f"pools={block['pool']}", flush=True)

    CHECK = 20
    spot_check = {
        name: {
            "positive": block["positive"][:CHECK],
            "negative": block["negative"][:CHECK],
        }
        for name, block in concepts.items()
    }

    payload = {
        "n_images": len(ids),
        "per_concept": PER_CONCEPT,
        "concepts": concepts,
        "spot_check": spot_check,
        "entropy_stats": {
            "min": float(min(entropy.values())),
            "median": float(np.median(list(entropy.values()))),
            "max": float(max(entropy.values())),
        },
        "source": "image content only; no audited-model scores used",
    }
    with open(os.path.join(OUT, "p3_concepts.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("[3] wrote p3_concepts.json")


if __name__ == "__main__":
    main()
