"""Merge the P4.2 qualitative exports into one countable table.

Input:  annotate/out/qualitative_*_labels.json (one file per annotator)
Output: refine-logs/p4_qualitative.json

The point of the manual pass is to replace "the ablation flips 38.8%" with a coded
account of what it removes when it flips: how often the deleted span is a slur or a
group reference, how often it is a stopword, and whether the annotator judged the flip
meaningful. Counts are split by flip / control so the control column can falsify the
story rather than decorate it.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

ANNOTATE = os.path.join(os.path.dirname(__file__), "..", "annotate")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
TAGS = ["slur", "negation", "link", "content", "function"]
VERDICTS = ["justified", "artefact", "unclear"]


def main() -> None:
    files = sorted(glob.glob(os.path.join(ANNOTATE, "out", "qualitative_*_labels.json")))
    if not files:
        raise SystemExit("no qualitative exports in annotate/out/")

    rows = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload.get("rows", []):
            row["annotator"] = row.get("annotator") or os.path.basename(path)
            rows.append(row)

    # one row per case: the first complete verdict wins, ties across annotators are
    # counted rather than averaged because these are categorical codes
    by_case = {}
    conflicts = 0
    for row in rows:
        key = row["id"]
        if key in by_case and by_case[key]["verdict"] != row["verdict"]:
            conflicts += 1
            continue
        by_case[key] = row

    table = defaultdict(Counter)
    for row in by_case.values():
        group = "flip" if row["flip"] else "control"
        table[group]["n"] += 1
        table[group][row["verdict"] or "unlabelled"] += 1
        for tag in row.get("tags", []):
            table[group][f"tag:{tag}"] += 1
        if row.get("note"):
            table[group]["with_note"] += 1

    summary = {
        "n_files": len(files),
        "n_rows": len(rows),
        "n_cases": len(by_case),
        "conflicts": conflicts,
        "groups": {group: dict(counts) for group, counts in table.items()},
        "tags": TAGS,
        "verdicts": VERDICTS,
        "criterion": "a flip counts as meaningful if the annotator marks 'justified' and the deleted span is a slur or group reference",
    }

    flips = table["flip"]
    controls = table["control"]
    summary["share_justified_flip"] = (flips["justified"] / flips["n"]) if flips["n"] else None
    summary["share_justified_control"] = (controls["justified"] / controls["n"]) if controls["n"] else None
    summary["share_slur_flip"] = (flips["tag:slur"] / flips["n"]) if flips["n"] else None
    summary["share_slur_control"] = (controls["tag:slur"] / controls["n"]) if controls["n"] else None

    with open(os.path.join(OUT, "p4_qualitative.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
