"""Fetch only the members we need from the remote MMHS150K archive.

The archive is 6 GB; the study uses 3,499 images. remotezip reads the central
directory once and pulls individual members with range requests.

Two modes:

  python scripts/fetch_subset.py                  draw a fresh stratified subset
  python scripts/fetch_subset.py --ids data/MMHS150K/subset_ids.json
                                                  reproduce the published subset

The committed `subset_ids.json` is the one every number in the report was computed on;
a freshly drawn subset is statistically comparable but not identical, so use --ids when
you want to match the report exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

URL = "http://datasets.cvc.uab.es/MMHS150K/MMHS150K.zip"
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")


def _stratum(entry: dict) -> tuple:
    labels = [int(x) for x in entry.get("labels", [0, 0, 0])]
    binary = int(sum(label > 0 for label in labels) >= 2)
    unique = len(set(labels))
    bucket = "all_agree" if unique == 1 else ("all_differ" if unique == 3 else "partial")
    return binary, bucket


def stratified_ids(gt: dict, n: int, seed: int = 0) -> list:
    import numpy as np

    rng = np.random.default_rng(seed)
    buckets = {}
    for tweet_id, entry in gt.items():
        buckets.setdefault(_stratum(entry), []).append(tweet_id)

    total = len(gt)
    picked = []
    for stratum, ids in buckets.items():
        quota = max(1, round(n * len(ids) / total))
        quota = min(quota, len(ids))
        chosen = rng.choice(len(ids), size=quota, replace=False)
        picked.extend(ids[i] for i in chosen)

    rng.shuffle(picked)
    picked = picked[:n]
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(picked)
    selection = picked[:500]
    print("  strata:", {f"{k[0]}/{k[1]}": len(v) for k, v in buckets.items()})
    return picked, selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--ids", default="",
                        help="reuse a committed subset_ids.json instead of re-sampling")
    args = parser.parse_args()
    os.makedirs(ROOT, exist_ok=True)

    from remotezip import RemoteZip

    started = time.time()
    with RemoteZip(URL) as archive:
        print(f"[1] archive indexed in {time.time() - started:.1f}s")

        names = set(archive.namelist())
        for name in ("MMHS150K_GT.json", "hatespeech_keywords.txt", "MMHS150K_readme.txt"):
            if name in names:
                with archive.open(name) as handle:
                    data = handle.read()
                with open(os.path.join(ROOT, name), "wb") as out:
                    out.write(data)
                print(f"    saved {name} ({len(data)} bytes)")

        for split in ("train", "val", "test"):
            member = f"splits/{split}_ids.txt"
            if member in names:
                os.makedirs(os.path.join(ROOT, "splits"), exist_ok=True)
                with archive.open(member) as handle:
                    data = handle.read()
                with open(os.path.join(ROOT, "splits", f"{split}_ids.txt"), "wb") as out:
                    out.write(data)

        with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
            gt = json.load(handle)
        print(f"[2] ground truth entries: {len(gt)}")

        if args.ids:
            with open(args.ids, encoding="utf-8") as handle:
                committed = json.load(handle)
            main_ids, selection = committed["main"], committed["selection"]
            print(f"[3] reusing committed ids: {len(main_ids)} main, {len(selection)} selection")
        else:
            main_ids, selection = stratified_ids(gt, args.n)
            print(f"[3] subset: {len(main_ids)} main, {len(selection)} selection")
        with open(os.path.join(ROOT, "subset_ids.json"), "w", encoding="utf-8") as handle:
            json.dump({"main": main_ids, "selection": selection}, handle)
        wanted = list(dict.fromkeys(list(main_ids) + list(selection)))

        image_dir = os.path.join(ROOT, "img_resized")
        ocr_dir = os.path.join(ROOT, "img_txt")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(ocr_dir, exist_ok=True)

        done = 0
        missing = 0
        for index, tweet_id in enumerate(wanted):
            member = f"img_resized/{tweet_id}.jpg"
            target = os.path.join(image_dir, f"{tweet_id}.jpg")
            if not os.path.exists(target):
                if member not in names:
                    missing += 1
                    continue
                try:
                    with archive.open(member) as handle:
                        data = handle.read()
                    with open(target, "wb") as out:
                        out.write(data)
                except Exception as error:
                    missing += 1
                    print(f"    fail {tweet_id}: {error}")
                    continue
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(wanted)} images, {time.time() - started:.0f}s", flush=True)

        print(f"[4] images: {done} saved, {missing} missing, {time.time() - started:.0f}s")

        ocr_saved = 0
        ocr_failed = 0
        for tweet_id in selection:
            member = f"img_txt/{tweet_id}.json"
            if member not in names:
                continue
            target = os.path.join(ocr_dir, f"{tweet_id}.json")
            if os.path.exists(target):
                continue
            try:
                with archive.open(member) as handle:
                    data = handle.read()
                with open(target, "wb") as out:
                    out.write(data)
                ocr_saved += 1
            except Exception as error:
                ocr_failed += 1
                print(f"    ocr fail {tweet_id}: {error}")
        print(f"[5] OCR files for selection set: {ocr_saved} ({ocr_failed} failed)")
    print(f"[6] finished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
