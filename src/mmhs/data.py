"""MMHS150K indexing, annotation statistics and stratified sampling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def normalize_labels(raw) -> tuple:
    """Coerce the annotation list to a tuple of ints (length varies: 1-5 voters)."""
    if raw is None:
        return (0,)
    if isinstance(raw, (int, float)):
        return (int(raw),)
    return tuple(int(x) for x in raw) or (0,)


@dataclass
class Sample:
    tweet_id: str
    text: str
    ocr_text: str
    image_path: str
    labels: Tuple[int, ...]

    @property
    def binary_strict(self) -> int:
        """Hate when a strict majority of annotators report a non-zero class."""
        votes = len(self.labels)
        non_zero = sum(label > 0 for label in self.labels)
        return int(non_zero * 2 > votes)

    @property
    def binary_loose(self) -> int:
        """Hate when any annotator reports a non-zero class."""
        return int(any(label > 0 for label in self.labels))

    @property
    def agreement(self) -> str:
        unique = set(self.labels)
        if len(unique) == 1:
            return "all_agree"
        if len(unique) == 3:
            return "all_differ"
        return "partial"


def load_ground_truth(root: str) -> Dict[str, dict]:
    for name in ("MMHS150K_GT.json", os.path.join("MMHS150K", "MMHS150K_GT.json")):
        path = os.path.join(root, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(f"MMHS150K_GT.json not found under {root}")


def load_split(root: str, split: str) -> Optional[List[str]]:
    for base in ("splits", os.path.join("MMHS150K", "splits")):
        path = os.path.join(root, base, f"{split}_ids.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]
    return None


def build_index(root: str, split: Optional[str] = None, require_image: bool = True) -> List[Sample]:
    gt = load_ground_truth(root)
    ids = load_split(root, split) if split else list(gt.keys())
    samples: List[Sample] = []
    for tweet_id in ids:
        entry = gt.get(str(tweet_id))
        if entry is None:
            continue
        image_path = _find_image(root, str(tweet_id))
        if require_image and image_path is None:
            continue
        samples.append(
            Sample(
                tweet_id=str(tweet_id),
                text=entry.get("tweet_text", ""),
                ocr_text=_read_ocr(root, str(tweet_id)),
                image_path=image_path or "",
                labels=normalize_labels(entry.get("labels")),
            )
        )
    return samples


def _find_image(root: str, tweet_id: str) -> Optional[str]:
    for base in ("img_resized", os.path.join("MMHS150K", "img_resized")):
        for ext in (".jpg", ".png", ".jpeg"):
            path = os.path.join(root, base, tweet_id + ext)
            if os.path.exists(path):
                return path
    return None


def _read_ocr(root: str, tweet_id: str) -> str:
    for base in ("img_txt", os.path.join("MMHS150K", "img_txt")):
        for ext in (".txt", ""):
            path = os.path.join(root, base, tweet_id + ext)
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    return handle.read().strip()
    return ""


def describe(samples: Sequence[Sample]) -> dict:
    max_votes = max(len(sample.labels) for sample in samples)
    labels = np.array([
        list(sample.labels) + [0] * (max_votes - len(sample.labels)) for sample in samples
    ])
    agreement = np.array([s.agreement for s in samples])
    return {
        "n": len(samples),
        "hate_strict": float(np.mean([s.binary_strict for s in samples])),
        "hate_loose": float(np.mean([s.binary_loose for s in samples])),
        "all_agree": float(np.mean(agreement == "all_agree")),
        "partial": float(np.mean(agreement == "partial")),
        "all_differ": float(np.mean(agreement == "all_differ")),
        "class_share": {
            name: float(np.mean(labels == idx))
            for idx, name in enumerate(["NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"])
        },
    }


def stratified_sample(
    samples: Sequence[Sample],
    n: int,
    seed: int = 0,
    key=lambda s: (s.binary_strict, s.agreement),
) -> List[Sample]:
    """Proportional stratified sampling on (binary label, annotator agreement)."""
    rng = np.random.default_rng(seed)
    buckets: Dict[tuple, List[Sample]] = {}
    for sample in samples:
        buckets.setdefault(key(sample), []).append(sample)

    total = len(samples)
    picked: List[Sample] = []
    for stratum, items in buckets.items():
        quota = max(1, round(n * len(items) / total))
        quota = min(quota, len(items))
        idx = rng.choice(len(items), size=quota, replace=False)
        picked.extend(items[i] for i in idx)

    if len(picked) > n:
        rng.shuffle(picked)
        picked = picked[:n]
    return picked
