"""Rebuild the analysis subsets from the official test split and fetch their images.

The first subset was stratified over the whole ground truth, so most samples may
have been in the audited model's training data. This script:

1. backs up the old subset ids,
2. stratifies 3,000 (plus a nested 500 selection set) over test_ids only,
3. fetches missing images through range requests on the remote archive.

Usage: python scripts/resplit_test_subset.py [--n 3000]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

URL = "http://datasets.cvc.uab.es/MMHS150K/MMHS150K.zip"
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "resplit_test_subset.log")


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def stratum(entry: dict) -> tuple:
    labels = [int(x) for x in entry.get("labels", [0])]
    votes = len(labels)
    binary = int(sum(label > 0 for label in labels) * 2 > votes)
    unique = len(set(labels))
    bucket = "all_agree" if unique == 1 else ("all_differ" if unique == votes else "partial")
    return binary, bucket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    args = parser.parse_args()

    subset_path = os.path.join(ROOT, "subset_ids.json")
    if os.path.exists(subset_path):
        backup = os.path.join(ROOT, "subset_ids_fullgt_backup.json")
        if not os.path.exists(backup):
            shutil.copy2(subset_path, backup)
            log("backed up old subset ids")

    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(ROOT, "splits", "test_ids.txt"), encoding="utf-8") as handle:
        test_ids = [line.strip() for line in handle if line.strip()]
    test_ids = [tweet_id for tweet_id in test_ids if tweet_id in gt]
    log(f"test split: {len(test_ids)} ids with annotations")

    buckets = {}
    for tweet_id in test_ids:
        buckets.setdefault(stratum(gt[tweet_id]), []).append(tweet_id)
    log("strata: " + str({f"{k[0]}/{k[1]}": len(v) for k, v in buckets.items()}))

    rng = np.random.default_rng(0)
    total = len(test_ids)
    picked = []
    for key, ids in buckets.items():
        quota = max(1, round(args.n * len(ids) / total))
        quota = min(quota, len(ids))
        chosen = rng.choice(len(ids), size=quota, replace=False)
        picked.extend(ids[i] for i in chosen)
    rng.shuffle(picked)
    picked = picked[:args.n]

    rng2 = np.random.default_rng(1)
    taken = set(picked)
    selection_pool = [tweet_id for tweet_id in test_ids if tweet_id not in taken]
    buckets_sel = {}
    for tweet_id in selection_pool:
        buckets_sel.setdefault(stratum(gt[tweet_id]), []).append(tweet_id)
    selection = []
    total_sel = len(selection_pool)
    for key, ids in buckets_sel.items():
        quota = max(1, round(500 * len(ids) / total_sel))
        quota = min(quota, len(ids))
        chosen = rng2.choice(len(ids), size=quota, replace=False)
        selection.extend(ids[i] for i in chosen)
    rng2.shuffle(selection)
    selection = selection[:500]

    with open(subset_path, "w", encoding="utf-8") as handle:
        json.dump({"main": picked, "selection": selection, "split": "test"}, handle)
    log(f"wrote new subset: {len(picked)} main, {len(selection)} selection (test split only)")

    from remotezip import RemoteZip

    image_dir = os.path.join(ROOT, "img_resized")
    os.makedirs(image_dir, exist_ok=True)
    wanted = list(dict.fromkeys(picked + selection))
    missing = [t for t in wanted if not os.path.exists(os.path.join(image_dir, f"{t}.jpg"))]
    log(f"images to fetch: {len(missing)} (already present: {len(wanted) - len(missing)})")
    if not missing:
        return

    started = time.time()
    with RemoteZip(URL) as archive:
        names = set(archive.namelist())
        done = 0
        absent = 0
        for tweet_id in missing:
            member = f"img_resized/{tweet_id}.jpg"
            if member not in names:
                absent += 1
                continue
            try:
                with archive.open(member) as handle:
                    data = handle.read()
                with open(os.path.join(image_dir, f"{tweet_id}.jpg"), "wb") as out:
                    out.write(data)
                done += 1
            except Exception as error:
                absent += 1
                log(f"fail {tweet_id}: {error}")
            if done % 200 == 0 and done:
                rate = done / max(time.time() - started, 1e-6)
                log(f"{done}/{len(missing)} fetched, {rate:.2f}/s, eta {(len(missing) - done) / rate / 60:.0f} min")

    log(f"done: {done} fetched, {absent} absent, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
