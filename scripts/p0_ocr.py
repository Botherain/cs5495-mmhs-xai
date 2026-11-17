"""P0.3: local OCR boxes for the 3,499 analysed images.

MMHS150K ships only the recognised text (img_txt/<id>.json, img_text) and no box
coordinates, so the boxes are produced here with EasyOCR (confidence >= 0.6), as
pre-registered in the proposal. Detections below the threshold are counted but
not used.

Outputs
  refine-logs/p0_ocr.jsonl       one line per image: boxes, text, confidence
  refine-logs/p0_ocr_stats.json  coverage, area share, excluded fraction

Usage: python scripts/p0_ocr.py            # resumable
       python scripts/p0_ocr.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
JSONL = os.path.join(OUT, "p0_ocr.jsonl")
MIN_CONF = 0.6


def load_ids() -> list:
    with open(os.path.join(ROOT, "subset_ids.json"), encoding="utf-8") as handle:
        subset = json.load(handle)
    return list(dict.fromkeys(subset["main"] + subset["selection"]))


def done_ids() -> set:
    if not os.path.exists(JSONL):
        return set()
    done = set()
    with open(JSONL, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-conf", type=float, default=MIN_CONF)
    args = parser.parse_args()

    import easyocr
    from PIL import Image

    ids = load_ids()
    finished = done_ids()
    todo = [tweet_id for tweet_id in ids if tweet_id not in finished]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[1] ids={len(ids)} done={len(finished)} todo={len(todo)}", flush=True)
    if not todo:
        print("[1] nothing to do", flush=True)
    else:
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        os.makedirs(OUT, exist_ok=True)
        started = time.time()
        with open(JSONL, "a", encoding="utf-8") as sink:
            for index, tweet_id in enumerate(todo, start=1):
                path = os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")
                record = {"id": tweet_id, "boxes": [], "text": "", "n_raw": 0}
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                    detections = reader.readtext(path, detail=1, paragraph=False)
                except Exception as error:
                    record["error"] = str(error)
                    detections = []
                    width = height = 0
                kept = []
                for box, text, conf in detections:
                    if conf < args.min_conf:
                        continue
                    xs = [float(point[0]) for point in box]
                    ys = [float(point[1]) for point in box]
                    kept.append(
                        {
                            "x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys),
                            "conf": round(float(conf), 4), "text": text,
                        }
                    )
                record.update(
                    {
                        "width": width,
                        "height": height,
                        "n_raw": len(detections),
                        "boxes": kept,
                        "text": " ".join(item["text"] for item in kept).strip(),
                    }
                )
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                if index % 100 == 0:
                    rate = index / max(time.time() - started, 1e-6)
                    print(
                        f"    {index}/{len(todo)} {rate:.2f}/s eta {(len(todo) - index) / rate / 60:.1f} min",
                        flush=True,
                    )

    summarise(ids, args.min_conf)


def summarise(ids: list, min_conf: float) -> None:
    records = {}
    with open(JSONL, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["id"]] = record

    covered = [record for record in records.values() if record.get("width")]
    with_text = [record for record in covered if record.get("boxes")]
    area_shares = []
    raw_counts = []
    kept_counts = []
    for record in covered:
        area = 0.0
        for box in record["boxes"]:
            area += max(box["x2"] - box["x1"], 0.0) * max(box["y2"] - box["y1"], 0.0)
        area_shares.append(min(area / max(record["width"] * record["height"], 1.0), 1.0))
        raw_counts.append(record.get("n_raw", 0))
        kept_counts.append(len(record["boxes"]))

    raw_total = sum(raw_counts)
    kept_total = sum(kept_counts)
    stats = {
        "requested": len(ids),
        "processed": len(covered),
        "with_text": len(with_text),
        "with_text_share": round(len(with_text) / max(len(covered), 1), 4),
        "boxes_raw": raw_total,
        "boxes_kept": kept_total,
        "excluded_fraction": round(1 - kept_total / max(raw_total, 1), 4),
        "mean_boxes_per_image": round(kept_total / max(len(covered), 1), 3),
        "mean_area_share": round(sum(area_shares) / max(len(area_shares), 1), 4),
        "median_area_share": round(sorted(area_shares)[len(area_shares) // 2], 4) if area_shares else None,
        "min_conf": min_conf,
        "engine": "easyocr-1.7.2 (en)",
        "missing": len(ids) - len(covered),
    }
    with open(os.path.join(OUT, "p0_ocr_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print("[2] " + json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
