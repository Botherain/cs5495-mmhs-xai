"""Merge the two manual annotation exports into the artefacts the report reads.

Inputs (produced by annotate/gold.html and annotate/concept.html, dropped into
annotate/out/):

  annotate/out/gold_labels.json     200 blind human judgements -> OCR gold standard
  annotate/out/concept_labels.json  1,824 concept-set images + 100 blind repeats

Outputs:

  refine-logs/p0_text_gold.json     EasyOCR detection / false-positive rates on the
                                    gold set, human-vs-OCR kappa, Wilson intervals
  refine-logs/p3_concept_text.json  inter-rater agreement on the 100 repeats and the
                                    measured error rate of the printed_text CAV
                                    positives/negatives

Both are optional for the notebook: if a file is missing the corresponding prose
degrades to "not measured" instead of inventing a number.

Usage: python scripts/merge_annotation.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "refine-logs")
ANN = os.path.join(ROOT, "annotate")
LABELS = {"text", "none", "unclear", "yes", "no"}


def read_rows(name: str) -> list:
    path = os.path.join(ANN, "out", name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    bad = [row for row in rows if row.get("label") not in LABELS]
    if bad:
        raise ValueError(f"{name}: {len(bad)} rows carry an unknown label, e.g. {bad[0]}")
    return rows


def read_all() -> tuple:
    """Every export in annotate/out/, resolved to one label per item.

    Several annotators is the normal case once the pages are shared, so the
    merge takes the majority per (queue, id, pass, concept) and reports the ties
    instead of silently letting the last file win.
    """
    import glob
    from collections import Counter, defaultdict

    files = sorted(glob.glob(os.path.join(ANN, "out", "*_labels.json")))
    votes = defaultdict(Counter)
    annotators = set()
    skipped = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload_rows = payload.get("rows", [])
        # The qualitative export writes tags and verdicts, not a single label, and
        # sits in the same directory. Skip it by schema rather than by filename so a
        # new queue cannot silently break the merge.
        if payload_rows and "label" not in payload_rows[0]:
            skipped.append(os.path.basename(path))
            continue
        for row in payload_rows:
            if row.get("label") not in LABELS:
                raise ValueError(f"{os.path.basename(path)}: unknown label {row.get('label')}")
            key = (row.get("queue", ""), row["id"], row.get("pass", 1), row.get("concept", ""))
            votes[key][row["label"]] += 1
            annotators.add(row.get("annotator") or os.path.basename(path))

    rows, ties = [], 0
    for (queue, tweet_id, pass_number, concept), counter in votes.items():
        ranked = counter.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            ties += 1
            continue
        rows.append({"id": tweet_id, "queue": queue, "pass": pass_number,
                     "concept": concept, "side": "", "label": ranked[0][0]})
    meta = {"n_files": len(files), "n_skipped": skipped, "n_annotators": len(annotators),
            "n_ties_dropped": ties, "annotators": sorted(annotators)}
    return rows, meta


def load_ocr() -> dict:
    records = {}
    with open(os.path.join(OUT, "p0_ocr.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["id"]] = record
    return records


def wilson(successes: int, total: int, z: float = 1.96) -> list:
    """Wilson score interval; binomial CIs at these n are not symmetric."""
    if total == 0:
        return [None, None]
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def kappa(pairs: list) -> float | None:
    """Cohen's kappa for two raters over paired categorical labels."""
    usable = [pair for pair in pairs if None not in pair]
    if not usable:
        return None
    n = len(usable)
    observed = sum(1 for a, b in usable if a == b) / n
    counts_a = Counter(a for a, _ in usable)
    counts_b = Counter(b for _, b in usable)
    expected = sum(counts_a[key] * counts_b[key] for key in set(counts_a) | set(counts_b)) / (n * n)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return (observed - expected) / (1 - expected)


def gold_report(rows: list, ocr: dict) -> dict:
    by_id = {}
    for row in rows:
        if row.get("queue") != "gold":
            continue
        # last write wins: the page lets a label be corrected
        by_id[row["id"]] = row["label"]

    usable = {tweet_id: label for tweet_id, label in by_id.items()
              if label in ("text", "none") and tweet_id in ocr}
    unclear = sum(1 for label in by_id.values() if label == "unclear")
    detected = {tweet_id: bool(ocr[tweet_id].get("boxes")) for tweet_id in usable}

    has_text = [tweet_id for tweet_id, label in usable.items() if label == "text"]
    no_text = [tweet_id for tweet_id, label in usable.items() if label == "none"]
    hits = sum(1 for tweet_id in has_text if detected[tweet_id])
    false_hits = sum(1 for tweet_id in no_text if detected[tweet_id])
    pairs = [(usable[tweet_id], "text" if detected[tweet_id] else "none") for tweet_id in usable]
    agreement = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None

    return {
        "n_labelled": len(by_id),
        "n_usable": len(usable),
        "n_unclear": unclear,
        "human_has_text_share": len(has_text) / len(usable) if usable else None,
        "ocr_detection_rate": hits / len(has_text) if has_text else None,
        "ocr_detection_ci95": wilson(hits, len(has_text)),
        "ocr_false_positive_rate": false_hits / len(no_text) if no_text else None,
        "ocr_false_positive_ci95": wilson(false_hits, len(no_text)),
        "ocr_precision": hits / (hits + false_hits) if (hits + false_hits) else None,
        "agreement_raw": agreement,
        "agreement_kappa": kappa(pairs),
        "n_human_text": len(has_text),
        "n_human_no_text": len(no_text),
        "criterion": "every gold image labelled; detection and false-positive rates reported with 95% Wilson intervals",
    }


def concept_report(rows: list, concepts: dict) -> dict:
    primary, repeat, seen = {}, {}, Counter()
    for row in rows:
        if row.get("queue") != "concept":
            continue
        if row.get("pass") == 2:
            repeat[row["id"]] = row["label"]
        else:
            primary[row["id"]] = row["label"]
        seen[row["id"]] += 1

    repeats = [(primary.get(tweet_id), label) for tweet_id, label in repeat.items()
               if primary.get(tweet_id) in ("text", "none") and label in ("text", "none")]
    raw = sum(1 for a, b in repeats if a == b) / len(repeats) if repeats else None

    def concept_rate(name: str, side: str, want: str) -> dict:
        ids = [tweet_id for tweet_id in concepts[name][side] if tweet_id in primary]
        hits = sum(1 for tweet_id in ids if primary[tweet_id] == want)
        return {"n": len(ids), "hits": hits,
                "rate": hits / len(ids) if ids else None,
                "ci95": wilson(hits, len(ids))}

    rates = {}
    for name, block in concepts.items():
        rates[name] = {
            # for the printed_text concept "want=text" is the expected label; for
            # face and flat_graphic the question is only whether text is present,
            # which is the covariate the CAV must not be learning.
            "positive_with_text": concept_rate(name, "positive", "text"),
            "negative_with_text": concept_rate(name, "negative", "text"),
            "n_positive_labelled": sum(1 for t in block["positive"] if t in primary),
            "n_negative_labelled": sum(1 for t in block["negative"] if t in primary),
        }

    return {
        "n_labelled": len(primary),
        "n_double": len(repeat),
        "n_double_usable": len(repeats),
        "double_agreement_raw": raw,
        "double_agreement_kappa": kappa(repeats),
        "criterion": "double-labelled agreement >= 0.8",
        "passes": bool(raw is not None and raw >= 0.8),
        "per_concept": rates,
    }


def membership_report(rows: list, manifest: dict) -> dict:
    """How well do the detector cues recover the concept they are standing in for?

    The sample is stratified 100 positive + 100 negative per concept, so the two
    quantities the design supports are sensitivity (share of cue-positives a human
    also calls positive) and specificity (share of cue-negatives a human rejects).
    Precision needs the population base rate, which the stratified sample cannot
    supply; it is deliberately not reported.
    """
    side_of = {(item["id"], item["concept"]): item["side"]
               for item in manifest.get("membership_items", [])}
    collected = {}
    for row in rows:
        if row.get("queue") != "membership":
            continue
        side = row.get("side") or side_of.get((row["id"], row.get("concept")), "")
        collected.setdefault(row.get("concept", ""), {}).setdefault(side, {})[row["id"]] = row["label"]

    per_concept = {}
    for name, block in collected.items():
        positives = [label for label in block.get("positive", {}).values() if label in ("yes", "no")]
        negatives = [label for label in block.get("negative", {}).values() if label in ("yes", "no")]
        hits = sum(1 for label in positives if label == "yes")
        misses = sum(1 for label in negatives if label == "no")
        per_concept[name] = {
            "n_positive_labelled": len(positives),
            "n_negative_labelled": len(negatives),
            "sensitivity": hits / len(positives) if positives else None,
            "sensitivity_ci95": wilson(hits, len(positives)),
            "specificity": misses / len(negatives) if negatives else None,
            "specificity_ci95": wilson(misses, len(negatives)),
        }

    return {
        "n_labelled": sum(len(block.get("positive", {})) + len(block.get("negative", {}))
                          for block in collected.values()),
        "per_concept": per_concept,
        "criterion": "the detector cue is usable if sensitivity and specificity both clear 0.8",
        "source": "concept set was built from detector cues, not from a human pass",
    }


def main() -> None:
    rows, meta = read_all()
    if not rows:
        print("no usable rows in annotate/out/ --- nothing to merge", file=sys.stderr)
        return
    print("[merge] " + json.dumps(meta, ensure_ascii=False))
    gold_rows = [row for row in rows if row["queue"] == "gold"]
    concept_rows = [row for row in rows if row["queue"] == "concept"]
    membership_rows = [row for row in rows if row["queue"] == "membership"]

    ocr = load_ocr()
    concepts = {}
    with open(os.path.join(OUT, "p3_concepts.json"), encoding="utf-8") as handle:
        concepts = json.load(handle)["concepts"]
    manifest = {}
    manifest_path = os.path.join(ANN, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

    if gold_rows:
        report = gold_report(gold_rows, ocr)
        with open(os.path.join(OUT, "p0_text_gold.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("[gold] " + json.dumps({key: report[key] for key in
                                      ("n_usable", "n_unclear", "ocr_detection_rate",
                                       "ocr_false_positive_rate", "agreement_kappa")}, indent=1))

    if concept_rows:
        report = concept_report(concept_rows, concepts)
        with open(os.path.join(OUT, "p3_concept_text.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("[concept] " + json.dumps({key: report[key] for key in
                                         ("n_labelled", "n_double", "double_agreement_raw",
                                          "double_agreement_kappa", "passes")}, indent=1))

    if membership_rows:
        report = membership_report(membership_rows, manifest)
        with open(os.path.join(OUT, "p3_concept_audit.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("[membership] " + json.dumps(report["per_concept"], indent=1))


if __name__ == "__main__":
    main()
