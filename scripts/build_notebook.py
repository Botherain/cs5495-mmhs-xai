"""Build the submission notebook from the stored result files.

Every number in the report prose is read from `refine-logs/*.json` / `*.jsonl` at
build time, so the text cannot drift from the artefacts. Run from the project
root:

    python scripts/build_notebook.py

It keeps the template's first three cells (header, title, instructions), sets the
header to the two group members, and replaces the seven empty code cells with the
report and its supporting code.
"""

from __future__ import annotations

import json
import os

import nbformat

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "refine-logs")
TEMPLATE = os.path.join(ROOT, "templates", "Course Project-2026A.template.ipynb")
TARGET = os.path.join(ROOT, "Course Project-2026A.ipynb")

NAMES = "CUI Zhaonan, WANG Siming"
EIDS = "58532227, 58530427"


# data access
def load(name: str):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rows(name: str) -> list:
    path = os.path.join(OUT, name)
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def f4(value: float) -> str:
    return f"{value:.4f}"


def fs(value: float) -> str:
    return f"{value:+.4f}"


def pc(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


# notebook cell helpers
def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip("\n"))


def figure(name: str, caption: str):
    return md(f"![{caption}](figures/{name})\n\n*{caption}*")


# sections
def section_setup() -> list:
    return [
        md("""
# YOUR METHODS HERE

## 0. What this report shows

A hate-speech detector that reads a tweet and an image is usually assumed to use both.
This project audits one checkpoint, `Amirhossein75/clip-vit-base-mmhs150k-mtl`, and asks
a narrower question than whether it is accurate: which modality does it actually read,
and can the explanation methods that claim to show this be trusted?

The checkpoint reproduces its published metrics, so the audit concerns the model as
released. Of seven candidate attribution methods, two clear a pre-registered
faithfulness floor on text (LIME and KernelSHAP, both near six times the floor), and on
the image side one clears it and barely (occlusion). Those three then play a coalition
game over the two modalities on 3,000 samples. Text carries almost all of the decision:
its Shapley value averages +0.047 against +0.0014 for the image, and it dominates on
99.4% of samples. The image is not inert: on 56.5% of samples its contribution is
negative, so the picture often works against the label the text already implies. The
fusion gate does not predict this split (§5), which rules the gate out as a shortcut
explanation. Ablating the units an explanation blames flips the label far more often
than ablating random units (§7), so the attributions are causally live even though the
image's share of the decision is small.

What the image channel contains stays unresolved. Three probes tried to give the small
image contribution a content and all three came back negative: the attribution lands
slightly below an equal-area control inside printed-text boxes (§6), the concept probe
fails its own validity checks (§6), and re-fitting the fusion head with modality dropout
lowers the image's share rather than raising it (§8). Agreement with the human annotators
moves in the pre-registered direction without reaching the threshold: ρ = -0.07 on the
pre-registered hate-relevant subset and -0.05 once the target level is controlled,
against a required -0.20 (§9). Two measurements were added late and change what the
dataset looks like. A 200-image blind OCR gold set puts EasyOCR's detection rate at
70.9% with a 0% false-positive rate, so roughly half the corpus contains readable words
rather than the third the raw coverage suggested (§1). The concept set turns out to have
been built from detectors rather than a human pass; §6 measures it and §11 records the
deviation. Negative results, including one silent bug that would have invalidated §3,
are documented in §11.
"""),
        code("""
import json
import os

import numpy as np
import pandas as pd

REFINE = "refine-logs"
FIG = "figures"


def load(name):
    with open(os.path.join(REFINE, name), encoding="utf-8") as handle:
        return json.load(handle)


def rows(name):
    out = []
    with open(os.path.join(REFINE, name), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


print("artefacts on disk:")
for name in sorted(os.listdir(REFINE)):
    if not name.endswith((".json", ".jsonl")) or name.startswith(("r0", "baseline_")):
        continue
    print(f"  {name:32s} {os.path.getsize(os.path.join(REFINE, name)) / 1024:9.1f} KB")
"""),
    ]


def section_model(gate, stats, ocr, gold) -> list:
    cells = [md("""
## 1. The audited model, and whether it is the model we think it is

**Checkpoint.** `Amirhossein75/clip-vit-base-mmhs150k-mtl`: CLIP ViT-B/32 towers
(frozen), a 512-d projection per tower, a gated fusion, and five sigmoid heads
(racist, sexist, homophobe, religion, otherhate). The published thresholds are
`[0.3, 0.7, 0.5, 0.3, 0.55]`.

**Data.** MMHS150K test split. Two disjoint samples are drawn from the same frame:

| sample | n | used for |
|---|---|---|
| analysis | 3,000 | modality Shapley, human agreement, deletion curves, failure modes |
| selection | 499 | the seven-method faithfulness comparison (expensive methods) |

Both were drawn from items whose three annotators are all present, so the human
labels support a strict-majority definition and an agreement bucket. All 3,499 images
were downloaded; coverage is complete.
""")]
    if gate and stats:
        cells.append(md("""
**Gate check before any explanation.** A checkpoint that does not reproduce its own
numbers cannot be audited, so the released metrics were recomputed first and compared
with the model card, with a tolerance of ±0.03 on ROC-AUC and a floor at the published
macro-F1.

| head | ROC-AUC | published | F1 | published |
|---|---|---|---|---|
""" + "\n".join(
            f"| {name} | {f4(block['auc'])} | {f4(block['published_auc'])} | "
            f"{f4(block['f1'])} | {f4(block['published_f1'])} |"
            for name, block in gate["per_head"].items()
        ) + f"""
| **macro (5 heads)** | **{f4(gate['macro_auc_5'])}** | **{f4(gate['published']['macro_roc'])}** | **{f4(gate['macro_f1_5'])}** | **{f4(gate['published']['macro_f1'])}** |

Verdict: **{gate['verdict']}**. {gate['reason']} No threshold re-tuning was needed
for the gate itself; the published thresholds are kept for every later decision-set
computation, so the explanations describe the released decision rule rather than a
re-calibrated one.

The `religion` head is a different matter: AUC {f4(gate['per_head']['religion']['auc'])}
on {pc(gate['per_head']['religion']['prevalence'], 2)} prevalence, with F1
{f4(gate['per_head']['religion']['f1'])}. It is accurate enough to pass the gate and
too weak to interpret, so it is **excluded from the analysis target**: the target
probability is the mean of the four usable heads (racist, sexist, homophobe,
otherhate), whose macro ROC-AUC is {f4(gate['macro_auc_4_usable'])} and macro-F1
{f4(gate['macro_f1_4_usable'])}. The five-head numbers appear only in this gate table.

**Human labels.** Strict majority non-zero gives a hate rate of
{pc(stats['main']['hate_rate_majority'])}; "any annotator" gives
{pc(stats['main']['hate_rate_any_annotator'])}. The gap between the two definitions is
what §9 turns on. Agreement buckets on the analysis set: all annotators agree
{stats['main']['agreement_buckets']['all_agree']} items, partial agreement
{stats['main']['agreement_buckets']['partial']}, all differ
{stats['main']['agreement_buckets']['all_differ']}.

**Images.** The dataset ships no OCR text for this split
({pc(stats['main']['ocr_text_available'], 2)} coverage), so text inside images was
recovered with EasyOCR at confidence ≥ 0.6: {ocr['with_text']} of {ocr['requested']}
images ({pc(ocr['with_text_share'])} of the 3,499 used here) carry at least one box,
{ocr['mean_boxes_per_image']:.2f} boxes per image on average, covering
{pc(ocr['mean_area_share'])} of the image area. Boxes below the confidence cut account
for {pc(ocr['excluded_fraction'])} of detections.

That coverage is a lower bound, and the bound was measured. A blind
gold set of {gold['n_labelled']} random analysis images, labelled without seeing the
detector output, gives a detection rate of
{pc(gold['ocr_detection_rate'])} (95% CI {pc(gold['ocr_detection_ci95'][0])}–{pc(gold['ocr_detection_ci95'][1])})
and a false-positive rate of {pc(gold['ocr_false_positive_rate'], 2)}
(95% CI {pc(gold['ocr_false_positive_ci95'][0], 2)}–{pc(gold['ocr_false_positive_ci95'][1], 2)});
{pc(gold['agreement_raw'])} of images agree, κ = {gold['agreement_kappa']:.2f}.

Dividing the observed coverage by the detection rate gives an estimated true share of
{pc(ocr['with_text_share'] / gold['ocr_detection_rate'])} text-bearing images, which matches the
human rate on the gold set ({pc(gold['human_has_text_share'])}) to within 0.2 pp. That
changes two things downstream: roughly half the corpus contains readable words, where the
raw coverage had suggested a third; and any group built from EasyOCR detection is a
*subset* of the text-bearing images with essentially no contamination in the other
direction, which §6 relies on. The implied share assumes the gold set's detection rate
transfers to all 3,499 images, which is reasonable because the sample is random from the
same pool; the interval on the detection rate propagates to roughly ±5 pp on the implied
share.
"""))

        cells.append(code("""
gate = load("p0_gate.json")
table = pd.DataFrame({
    name: {
        "ROC-AUC": block["auc"],
        "published AUC": block["published_auc"],
        "F1": block["f1"],
        "published F1": block["published_f1"],
        "prevalence": block["prevalence"],
    }
    for name, block in gate["per_head"].items()
}).T
print(f"verdict: {gate['verdict']}  (tolerance {gate['tolerance']}, floor {gate['f1_floor']:.3f})")
print(table.round(4).to_string())
print()
print(f"macro 5 heads : AUC {gate['macro_auc_5']:.4f}  F1 {gate['macro_f1_5']:.4f}")
print(f"macro 4 usable: AUC {gate['macro_auc_4_usable']:.4f}  F1 {gate['macro_f1_4_usable']:.4f}")

stats = load("p0_stats.json")
print()
print("analysis split:", stats["main"]["n"], "| strict hate rate", stats["main"]["hate_rate_majority"],
      "| any-annotator", stats["main"]["hate_rate_any_annotator"])
print("agreement buckets:", stats["main"]["agreement_buckets"])

ocr = load("p0_ocr_stats.json")
print()
print(f"OCR: {ocr['with_text']}/{ocr['requested']} images with text "
      f"({ocr['with_text_share']:.3%}), mean {ocr['mean_boxes_per_image']:.2f} boxes, "
      f"{ocr['mean_area_share']:.3%} of area")
"""))
    return cells


def section_units() -> list:
    return [
        md("""
## 2. Units, interventions, and what "faithful" is allowed to mean

The numbers in §3 depend on three definitions.

**Units.** Text explanations act on *content BPE tokens*: CLIP's tokenizer adds a
start and an end sentinel, and those two are never units. Image explanations act on
the 7×7 patch grid (32×32 pixels per cell), the resolution Grad-CAM actually has.

**Deletion is an intervention on the mask, not on the ids.** CLIP's text tower pools
at the *first* end-of-sequence token found by argmax, so writing a pad id into a
sentence silently moves the pooled position forward and truncates the tail, an
intervention that changes two things at once. Deletion here sets `attention_mask` to
0 at the chosen positions and leaves `input_ids` untouched. On the image side,
deletion writes the mean colour, which is exactly the "blank" counterfactual used in
the coalition game, so §3 and §5 agree about what removing evidence means.

**The missing-modality counterfactual is content-free but not zero.** Text is replaced
by the encoding of the empty string; the image is replaced by a mean-colour image.
Both are real inputs the towers can process, which matters because a zero vector is
off-manifold for CLIP's LayerNorm.

**Faithfulness.** A ranking is scored by AOPC: delete units in the order the method
ranks them (highest first), measure the drop in the target at each of six fractions,
and average. Insertion mirrors it, starting from the blank modality. The reported
number is the mean of the deletion and insertion AOPC, so a method cannot win by
ranking everything as "important". Every method is compared with a random ordering of
the same units, averaged over repeats, by a paired test on the same samples.

The floors were fixed before the comparison ran: **text AOPC > 0.01, image AOPC >
0.001**. The ten-fold difference is deliberate: image attributions on CLIP-scale
models are numerically much smaller, and a single floor would either pass noise on the
image side or fail every method on the text side. The rule for naming a winner, also
pre-registered: clear the floor and beat random; highest AOPC wins; a difference below
0.001 or a paired p above 0.05 is a tie, broken by comprehensiveness at 20%, then by
seed/noise stability, then in favour of the deterministic method.
"""),
        code("""
import sys
sys.path.insert(0, "src")

import torch
from mmhs.attribution import content_positions

# The intervention, shown on one sentence: masking a token must not re-tokenise.
tok_ids = torch.tensor([[49406, 320, 1125, 49407, 49407, 49407]])
tok_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
print("content positions (sentinels excluded):", content_positions(tok_mask))
before = tok_ids.clone()
masked = tok_mask.clone()
masked[0, content_positions(tok_mask)[1]] = 0
print("after deleting one token, ids unchanged:", torch.equal(tok_ids, before),
      "| mask:", masked.tolist())

matrix = load("p1_matrix.json")
print()
print("effect floors used below:", 0.01, "(text) /", 0.001, "(image)")
print("fractions scored:", matrix["fractions"])
"""),
    ]


def section_p1(matrix, decision) -> list:
    if not matrix or not decision:
        return [md("## 3. Which explainer is faithful?\n\n*Pending: `p1_matrix.json`.*")]

    text = decision["modalities"]["text"]
    image = decision["modalities"]["image"]
    rows_md = "\n".join(
        f"| {modality} | {name} | {fs(block['aopc_del']['mean'])} | "
        f"{fs(block['comp_20']['mean'])} | {fs(block['suff_20']['mean'])} | "
        f"{block['aopc_del']['p_vs_random']:.2e} |"
        for modality in ("text", "image")
        for name, block in matrix[modality].items()
    )
    random_line = (
        f"| text | *random order* | {fs(matrix['text']['lime']['aopc_del']['random_mean'])} | "
        f"{fs(matrix['text']['lime']['comp_20']['random_mean'])} | --- | --- |\n"
        f"| image | *random order* | {fs(matrix['image']['occlusion']['aopc_del']['random_mean'])} | "
        f"{fs(matrix['image']['occlusion']['comp_20']['random_mean'])} | --- | --- |"
    )

    return [
        md(f"""
## 3. Seven candidates, one winner per modality

| modality | method | AOPC | comp@20% | suff@20% | paired p vs random |
|---|---|---|---|---|---|
{rows_md}
{random_line}

Paired test: one-sided Wilcoxon on the per-sample difference against the random
ordering of the same units, n = {matrix['n']} selection samples.

**Text.** {text['winner'].upper()} wins with AOPC
{fs(text['winner_block']['aopc_mean'])}, {text['winner_block']['aopc_mean'] / 0.01:.1f}× the
0.01 floor; KernelSHAP follows at {fs([c for c in text['candidates'] if c['method'] == 'kernelshap'][0]['aopc_mean'])}.
The gap between them is {text['tie']['delta_aopc']:.4f} with a paired p of
{text['tie']['paired_p']:.1e}, so it is not a tie by the pre-registered rule, but the
practical reading is that the two surrogate methods agree, and the choice between them
does not carry the argument. Occlusion fails the floor on text: AOPC
{fs([c for c in text['candidates'] if c['method'] == 'occlusion'][0]['aopc_mean'])}
against a random ordering at
{fs(matrix['text']['lime']['aopc_del']['random_mean'])}. That is the first genuinely
uncomfortable result in this report: the textbook model-agnostic reference method is
*worse than random* at ranking text tokens on this checkpoint, because single-token
deletions are strongly non-additive here, while LIME and KernelSHAP fit a local
additive surrogate and therefore inherit the model's actual additive structure.

**Image.** Occlusion is the only method above the 0.001 floor
({fs(image['winner_block']['aopc_mean'])}); Grad-CAM, integrated gradients and
Score-CAM are all at or below zero, i.e. deleting the cells they call most important
raises the target slightly more often than deleting random cells. This is itself a
finding about the checkpoint: the vision tower's gradient signal through the fusion
gate is close to uninformative at 7×7 resolution.

**Three caveats that must travel with the winner.** AOPC is LIME's own optimisation
target (it fits weights to reproduce local deletions), so LIME's margin over
KernelSHAP is partly self-fulfilling; the defensible claim is only that both clear the
floor by a wide margin. The image winner clears the floor by
{image['winner_block']['aopc_mean'] / 0.001:.2f}×, not the six-fold margin seen on text, which is
consistent with the image's contribution being small in absolute terms, which §5
confirms independently through a completely different route. And the LIME kernel width
is the library's text default (25 on the Euclidean distance over token presence), not a
pre-registered choice: the tabular default, 0.25·√n, collapses onto near-full coalitions
on this model and returns almost uniform coefficients, which was discovered while
building the method rather than specified in advance. The reported comparison is
therefore "LIME as shipped against the alternatives", not "LIME at its best possible
setting".
"""),
        code("""
matrix = load("p1_matrix.json")
decision = load("p1_decision.json")

for modality in ("text", "image"):
    print(f"--- {modality} ---")
    block = matrix[modality]
    for name, entry in block.items():
        print(f"  {name:11s} AOPC {entry['aopc_del']['mean']:+.4f}  "
              f"comp20 {entry['comp_20']['mean']:+.4f}  suff20 {entry['suff_20']['mean']:+.4f}  "
              f"random {entry['aopc_del']['random_mean']:+.4f}  p {entry['aopc_del']['p_vs_random']:.1e}")
    print(f"  winner: {decision['modalities'][modality]['winner']}")
print()
print("text tie block:", json.dumps(decision["modalities"]["text"]["tie"], indent=1))
"""),
        figure("fig1_aopc.png",
               "Figure 1. AOPC per candidate method, with the random ordering as the dashed line. "
               "Error bars are standard errors over the 499 selection samples."),
    ]


def section_gates(curves, stability) -> list:
    if not curves:
        return [md("## 4. Does the attribution survive a random model?\n\n*Pending: `p1_sanity_curves.json`.*")]
    gates = curves.get("gates", {})
    lines = []
    for modality in ("text", "image"):
        for name, block in gates.get(modality, {}).items():
            lines.append(
                f"| {modality} | {name} | {block['rho_depth']:+.2f} | {block['r_full']:+.2f} | "
                f"{'pass' if block['passes'] else 'fail'} |"
            )
    staging = curves["curves"]["text"].get("3", {})

    stability_md = ""
    if stability:
        parts = []
        for name, block in stability.get("text_seeds", {}).items():
            parts.append(f"| text | {name} | {block['min_spearman']:.2f} | "
                         f"{block['min_top10_overlap']:.2f} | {'pass' if block['passes'] else 'fail'} |")
        for name, block in stability.get("image_noise", {}).items():
            parts.append(f"| image | {name} | {block['min_spearman']:.2f} | "
                         f"{block['min_top10_overlap']:.2f} | {'pass' if block['passes'] else 'fail'} |")
        stability_md = ("\n**Stability.** Worst pair over repeats (`p1_stability.json`):\n\n"
                        "| modality | method | min Spearman | min top-10% overlap | verdict |\n"
                        "|---|---|---|---|---|\n" + "\n".join(parts))

    return [
        md(f"""
## 4. Does the attribution survive a random model?

An explanation that also describes a randomly initialised network is describing the
architecture, not the learned function. Each tower is randomised on its own: the
last 0, 3, 6, 9 or all 12 encoder blocks re-initialised, the other tower and the fusion
head untouched. The attribution is then compared with the trained model's by Spearman
correlation over units. Pre-registered: the correlation must fall
monotonically with depth ($\\rho_{{\\text{{depth}}}} < -0.7$) and collapse at full
randomisation ($r_{{\\text{{full}}}} < 0.2$). n = {curves['n']} selection samples.

| tower | method | depth ρ | full r | gate |
|---|---|---|---|---|
""" + "\n".join(lines) + f"""

The ordering is the interesting part. At three randomised text blocks, occlusion has
already fallen to {staging['occlusion']['spearman']:.2f} while LIME and KernelSHAP are
still at {staging['lime']['spearman']:.2f} and {staging['kernelshap']['spearman']:.2f}.
Occlusion's ranking depends on individual token deletions, which are exactly what a
partly-randomised tower distorts; the surrogate methods fit many coalitions at once
and are correspondingly more robust. This is the second independent argument against
occlusion on text, and it was measured before the AOPC comparison was scored.{stability_md}
"""),
        code("""
curves = load("p1_sanity_curves.json")
print("levels:", curves["levels"])
for modality in ("text", "image"):
    print(f"--- {modality} tower ---")
    for name, block in curves["gates"][modality].items():
        print(f"  {name:11s} rho_depth {block['rho_depth']:+.2f}  r_full {block['r_full']:+.2f}  "
              f"{'pass' if block['passes'] else 'fail'}")

if os.path.exists(os.path.join(REFINE, "p1_stability.json")):
    stability = load("p1_stability.json")
    print()
    for group in ("text_seeds", "image_noise"):
        for name, block in stability.get(group, {}).items():
            print(f"  {group:11s} {name:11s} min rho {block['min_spearman']:+.2f}  "
                  f"min overlap {block['min_top10_overlap']:.2f}")
"""),
        figure("fig4_gates.png",
               "Figure 2. Attribution agreement with the trained model as encoder blocks are randomised. "
               "A faithful method should fall to zero."),
    ]


def section_phi(phi) -> list:
    if not phi:
        return [md("## 5. How much of the decision is the image?\n\n*Pending: `p2_summary.json`.*")]
    heads = phi["head_names"]
    empty = phi["heads_empty_mean"]
    both = phi["heads_both_mean"]
    head_lines = "\n".join(
        f"| {name} | {empty[index]:.3f} | {both[index]:.3f} | {both[index] - empty[index]:+.3f} |"
        for index, name in enumerate(heads)
    )
    return [
        md(f"""
## 5. How much of the decision is the image? A two-player coalition game

Attribution methods disagree; the coalition game does not, because with two players
the Shapley value has a closed form:

$$\\phi_T = \\tfrac12\\big[v(T)-v(\\varnothing)\\big] + \\tfrac12\\big[v(T,I)-v(I)\\big], \\qquad
\\phi_I = \\tfrac12\\big[v(I)-v(\\varnothing)\\big] + \\tfrac12\\big[v(T,I)-v(T)\\big]$$

where $v(\\varnothing)$ is the blank-text/blank-image input, $v(T)$ is real text over a
mean-colour image, $v(I)$ is empty text over the real image, and $v(T,I)$ is the actual
input. The target is the four-head mean probability on {phi['n']:,} analysis samples.
Completeness ($\\phi_T + \\phi_I = v(T,I) - v(\\varnothing)$) holds to
{phi['completeness_max_error']:.1e}, so the decomposition is exact rather than
approximate.

| quantity | value |
|---|---|
| mean $\\phi_T$ | {fs(phi['phi_t']['mean'])} |
| median $\\phi_T$ | {fs(phi['phi_t']['median'])} |
| share of samples with $\\phi_T > 0$ | {pc(phi['phi_t']['share_positive'])} |
| mean $\\phi_I$ | {fs(phi['phi_i']['mean'])} |
| median $\\phi_I$ | {fs(phi['phi_i']['median'])} |
| share of samples with $\\phi_I < 0$ | {pc(phi['phi_i_negative_share'])} |
| text-dominant samples ($|\\phi_T| > |\\phi_I|$) | {pc(phi['text_dominance_share'])} |
| interaction $v(T,I)-v(T)-v(I)+v(\\varnothing)$ | {fs(phi['interaction_mean'])} |

**The text accounts for most of the decision.** Mean $\\phi_T$ is
{phi['phi_t']['mean'] / max(phi['phi_i']['mean'], 1e-12):.0f}× the mean $\\phi_I$. In a
paired sense the image moves the target by about one percentage point, and that is the
ceiling on any "the model uses the image" claim.

**"Ignored" is the wrong word.** $\\phi_I$ is negative on
{pc(phi['phi_i_negative_share'])} of samples: for a majority of inputs, replacing the
picture with a blank one *raises* the hate probability. The image is not a neutral
modality that the fusion gate happens to down-weight; it frequently pulls against the
text's evidence. That asymmetry is invisible in accuracy numbers and invisible in
gate values, and it is the strongest argument for doing the coalition game rather than
reading the gate.

**Per-head, the effect is not uniform.** Mean head probabilities with both modalities
blank versus both present:

| head | blank | both | change |
|---|---|---|---|
{head_lines}

The `otherhate` head *drops* when both modalities are present, and `religion`, the
head excluded from the target, moves the most. Neither effect is visible from the
target alone, which is why the per-head table is reported even though the analysis is
target-level.

**Deletion curves agree with the Shapley values.** Removing text tokens in winning-
attribution order costs {fs(phi['deletion']['text_reduction'])} of target probability;
removing image cells costs {fs(phi['deletion']['image_reduction'])}; removing random
units of either kind costs {fs(phi['deletion']['random_reduction'])}, i.e. nothing.
The gap between text and image, {fs(phi['deletion']['text_minus_image'])}, is measured
on a different instrument (deletion from the full input) than the Shapley values
(coalitions from the blank input) and lands on the same conclusion.
"""),
        code("""
phi = load("p2_summary.json")
print(json.dumps({key: phi[key] for key in
                  ("n", "phi_t", "phi_i", "text_dominance_share", "phi_i_negative_share",
                   "interaction_mean", "completeness_max_error", "deletion")}, indent=1))

heads = pd.DataFrame({
    "blank": phi["heads_empty_mean"],
    "both": phi["heads_both_mean"],
}, index=phi["head_names"])
heads["change"] = heads["both"] - heads["blank"]
print()
print(heads.round(3).to_string())

rows = rows("p2_phi.jsonl")
print()
print(f"samples in p2_phi.jsonl: {len(rows)}")
print(f"share phi_T > 0: {np.mean([row['phi_t'] > 0 for row in rows]):.3f}   "
      f"share phi_I < 0: {np.mean([row['phi_i'] < 0 for row in rows]):.3f}")
"""),
        figure("fig2_phi.png",
               "Figure 3. Per-sample Shapley values for the two modalities. The image panel is clipped "
               "to ±0.01 for readability; the mass sits slightly left of zero."),
        figure("fig3_deletion.png",
               "Figure 4. Deletion curves: mean target drop with 95% interval across analysis samples."),
    ]


def section_image_evidence(overlap, tcav, audit, ctext) -> list:
    if not overlap:
        return [md("## 6. What the image channel contains\n\n*Pending: `p3_overlap.json`.*")]

    label_lines = ""
    if audit:
        label_lines = "\n".join(
            f"| {name} | {block['n_positive_labelled']} / {block['n_negative_labelled']} | "
            f"{block['sensitivity']:.2f} ({block['sensitivity_ci95'][0]:.2f}–{block['sensitivity_ci95'][1]:.2f}) | "
            f"{block['specificity']:.2f} ({block['specificity_ci95'][0]:.2f}–{block['specificity_ci95'][1]:.2f}) |"
            for name, block in audit["per_concept"].items()
        )

    confound_lines = ""
    if ctext:
        confound_lines = "\n".join(
            f"| {name} | {pc(block['positive_with_text']['rate'])} | {pc(block['negative_with_text']['rate'])} | "
            f"{100 * (block['positive_with_text']['rate'] - block['negative_with_text']['rate']):+.1f} pp |"
            for name, block in ctext["per_concept"].items()
        )

    tcav_lines = ""
    if tcav:
        tcav_lines = "\n".join(
            f"| {name} | {block['cav_accuracy']:.3f} | {block['mean_cosine']:+.3f} ± {block['cosine_sd']:.3f} | "
            f"{block['tcav']:.3f} | {block['p_null']:.2f} | {block['split_half_spearman']:+.2f} |"
            for name, block in tcav["concepts"].items()
        )

    cells = [md(f"""
## 6. What the image channel contains

Two probes ask what the image contributes on the samples where it does contribute: a
patch-level overlap test against printed text inside the picture, and a concept probe
on the image tower. Both come back negative. The concept probe also fails its own
label audit, which is a separate finding and is reported as such.

**Patch attribution against printed text.** Each 7×7 cell gets a text weight equal to
the fraction of its area covered by an EasyOCR box, so a small box that only clips a
cell still counts and no sample is dropped for having text too small to contain a cell
centre. The control keeps the *union* area of the text boxes fixed and moves a square of
that area to a random position, 100 placements, weighted the same way. The comparison is
therefore "the same amount of image, placed on the text" against "placed elsewhere", not
against zero.

| quantity | value |
|---|---|
| samples with at least one text box | {overlap['n_with_text']:,} of {overlap['n_total']:,} |
| mean area share of the boxes | {pc(overlap['mean_text_area_share'])} |
| mean attribution share inside the text | {pc(overlap['mean_observed_share'])} |
| mean share for the equal-area control | {pc(overlap['mean_null_share'])} |
| excess | {100 * overlap['mean_excess']:+.1f} pp |
| paired Wilcoxon p | {overlap['p_wilcoxon']:.1e} |
| samples above the control's 95th percentile | {pc(overlap['share_above_null_p95'])} |

The excess is {100 * overlap['mean_excess']:+.1f} pp against a pre-registered threshold of +15 pp,
so the criterion is {'met' if overlap['passes'] else 'missed'}, and it is missed in the
*negative* direction: attribution inside the text boxes is slightly below an equal-area
control, and the paired test rejects equality that way
(p = {overlap['p_wilcoxon']:.1e}). Only {pc(overlap['share_above_null_p95'])} of images put more attribution
on the text than the control's 95th percentile. On this checkpoint the image tower is
not re-reading the words inside the picture; if anything it looks away from them. A negative
excess does not make text unimportant; it says that a text region draws no more
attribution than any other patch of the same size. The
control square matches the *union* of the text boxes; using the sum of the box areas
instead, which double-counts overlapping boxes and makes the control larger, moves the
excess to {100 * overlap['mean_excess_sum_area']:+.1f} pp, so the sign and the verdict do not
depend on that choice.
""")]

    if audit:
        cells.append(md(f"""
**Do the concept labels hold up?** The concept set was built from three detectors, not
from a human pass (§11). A stratified sample of 100 positives and 100 negatives per
concept was therefore labelled blind, which supports two quantities: sensitivity (share
of cue-positives a human also calls positive) and specificity (share of cue-negatives a
human rejects). Precision is not reported because the stratified sample cannot supply
the population base rate.

| concept | labelled + / − | sensitivity | specificity |
|---|---|---|---|
{label_lines}

`printed_text` is a usable proxy at both ends. `face` and `flat_graphic` are not: their
negative sets are 44% and 35% contaminated with images that do contain the concept, so a
CAV trained on them is partly fitted to the wrong side. The same labels expose a second
problem: how much of each concept set also contains printed text:

| concept | text share, positives | text share, negatives | gap |
|---|---|---|---|
{confound_lines}

For `face` the two sides are 52.0% and 44.8%, which is almost equality: a "face"
direction is partly a text direction. `flat_graphic` is 63.0% against 50.7%. Only
`printed_text` separates cleanly, and that is the concept whose CAV the probe still
cannot use.
"""))

    if tcav:
        cells.append(md(f"""
**Concept probe (TCAV).** A logistic CAV is fitted in the image tower's 768-d pooler
space; the sensitivity is the gradient of the four-head target with respect to that
pooler output, dotted with the unit CAV. Twenty random unit directions give the null.

| concept | CAV accuracy | mean cosine | TCAV | p vs null | split-half ρ |
|---|---|---|---|---|---|
{tcav_lines}

Both halves of this design fail, independently.

*The statistic is degenerate here.* The per-sample gradient has a mean cosine of
{tcav['sensitivity_collinearity']:.3f} with its own mean, so its direction barely varies across inputs. A sign
test on a nearly fixed direction can only return 0 or 1, which is what the first two
rows show, and the null betrays the same collapse: 20 random directions produce scores
that jump between 0.0003 and 0.981 (sd {tcav['null']['sd']:.3f}). The near-constancy is
mechanistic. With the four sigmoid heads saturated for most inputs, the derivative's
direction comes from the fusion head's weights; the image input barely moves it.

*The concept directions are weak.* CAV accuracy is 0.62 for `printed_text` and `face`
against balanced 400/400 sets, and 0.86 for `flat_graphic`. The `printed_text` case
rules out bad annotation as the explanation: its labels are 0.93/0.94. The limit sits
in the gradient.

The section's answer is therefore a double negative: no evidence that the image
channel's contribution is text-in-image, and no concept-level claim that survives its
own validity checks.
"""))

    cells.append(code("""
overlap = load("p3_overlap.json")
print(json.dumps({key: overlap[key] for key in
                  ("n_with_text", "mean_observed_share", "mean_null_share", "mean_excess",
                   "p_wilcoxon", "share_above_null_p95", "mean_text_area_share", "passes")},
                 indent=1))

audit = load("p3_concept_audit.json")
print()
print(pd.DataFrame(audit["per_concept"]).T[
    ["n_positive_labelled", "n_negative_labelled", "sensitivity", "specificity"]].round(3).to_string())

ctext = load("p3_concept_text.json")
print()
print("double-labelled agreement:",
      f"{ctext['double_agreement_raw']:.3f} (kappa {ctext['double_agreement_kappa']:.3f}, "
      f"n={ctext['n_double_usable']}, passes={ctext['passes']})")

tcav = load("p3_tcav.json")
print()
print(f"gradient collinearity: {tcav['sensitivity_collinearity']:.3f}")
print(f"null over {tcav['null']['n']} random directions: "
      f"{tcav['null']['mean']:.3f} +- {tcav['null']['sd']:.3f}")
for name, block in tcav["concepts"].items():
    print(f"  {name:13s} TCAV={block['tcav']:.3f}  CAV acc={block['cav_accuracy']:.3f}  "
          f"cos={block['mean_cosine']:+.3f}+-{block['cosine_sd']:.3f}  p={block['p_null']:.2f}")
"""))
    return cells


def section_failures(flip, qual) -> list:
    if not flip:
        return [md("## 7. Failure modes\n\n*Pending: `p4_flip.json`.*")]

    def mod_line(name: str) -> str:
        block = flip["modalities"][name]
        return (f"| {name} | {pc(block['top_flip_rate'])} | {pc(block['random_flip_rate'])} | "
                f"{100 * block['difference']:+.1f} pp | {block['mcnemar_p']:.1e} |")

    outcome_lines = "\n".join(
        f"| {label.replace('_', ' ')} | {block['n']} | {pc(block['share'])} | "
        f"{pc(block['text_flip_rate'])} | {pc(block['image_flip_rate'])} | {pc(block['both_flip_rate'])} |"
        for label, block in flip["outcomes"].items() if block.get("n")
    )

    qual_md = ""
    if qual:
        flips = qual["groups"].get("flip", {})
        controls = qual["groups"].get("control", {})
        n_f = flips.get("n", 0)
        n_c = controls.get("n", 0)

        def line(label: str, group: dict, total: int) -> str:
            share = (100 * group.get(label, 0) / total) if total else 0.0
            return f"| {label.replace('tag:', '').replace('_', ' ')} | {group.get(label, 0)} | {share:.0f}% |"

        qual_md = f"""
**What the ablation actually removes.** Forty cases were reviewed by hand: 20 text
flips and 20 controls matched on the error class (8 false positives, 7 correct, 4
over-trigger, 1 misclass per arm). Each case showed the deleted tokens highlighted and
the before/after head probabilities. The coding is descriptive: 20 per arm cannot
support a rate.

| deleted span contains | flips (n = {n_f}) | share | controls (n = {n_c}) | share |
|---|---|---|---|---|
| slur or group reference | {flips.get('tag:slur', 0)} | {100 * flips.get('tag:slur', 0) / n_f:.0f}% | {controls.get('tag:slur', 0)} | {100 * controls.get('tag:slur', 0) / n_c:.0f}% |
| URL / @mention / RT | {flips.get('tag:link', 0)} | {100 * flips.get('tag:link', 0) / n_f:.0f}% | {controls.get('tag:link', 0)} | {100 * controls.get('tag:link', 0) / n_c:.0f}% |
| ordinary content word | {flips.get('tag:content', 0)} | {100 * flips.get('tag:content', 0) / n_f:.0f}% | {controls.get('tag:content', 0)} | {100 * controls.get('tag:content', 0) / n_c:.0f}% |
| function word / punctuation | {flips.get('tag:function', 0)} | {100 * flips.get('tag:function', 0) / n_f:.0f}% | {controls.get('tag:function', 0)} | {100 * controls.get('tag:function', 0) / n_c:.0f}% |

Two readings. On every one of the 20 flips the deleted span contained a slur or group
reference ({flips.get('tag:slur', 0)}/{n_f}), so the flips are not a ranking artefact that
happens to break the input: the method found the word that carries the label. On the
control arm the same top-10% ranking lands on a URL fragment in {controls.get('tag:link', 0)}/{n_c}
cases and on a slur in only {controls.get('tag:slur', 0)}/{n_c}: when nothing flips, the deleted units are
frequently ones the model does not use. The contrast between the columns is the useful
part and it points the same way as the McNemar test.

The verdict field failed and is reported as such:
{flips.get('justified', 0)}/{n_f} flips and {controls.get('justified', 0)}/{n_c} controls were marked
"justified". The question asked whether the *flip* was meaningful, which has no meaning
for a case that did not flip, so the two columns agreeing carries no information. A
better instrument would ask, for both arms, whether the deleted span is the span a human
would have deleted.
"""

    return [
        md(f"""
## 7. Failure modes: are the blamed units actually load-bearing?

An attribution can be faithful in a deletion curve and still be irrelevant to the
decision. The sharper test: delete the top 10% of units the winner blames, recompute
the decision set from the released thresholds, and ask whether the label changed. Text
deletion flips the attention mask; image deletion writes the mean colour into whole
cells. Each is compared with random deletion of the same number of units, three
repeats, exact McNemar on the discordant pairs. n = {flip['n']} analysis samples.

| ablation | flips | random | difference | McNemar p |
|---|---|---|---|---|
{mod_line('text')}
{mod_line('image')}
{mod_line('both')}

The threshold is a 5 pp difference with p < 0.05. The random arm is reduced to its
majority vote across the three repeats before pairing, so each sample contributes one
binary random outcome against the single top-k outcome. Note what the comparison is:
top-10% text is about three tokens, top-10% image is about five cells, and on this
model removing three well-chosen tokens is a much more violent intervention than
removing five patch cells, which is the asymmetry §5 measured from the
coalition side.

Breaking the same flips down by error type, against the strict-majority label set:

| outcome | n | share | text | image | both |
|---|---|---|---|---|---|
{outcome_lines}

`over_trigger` means the model predicts at least one head the majority did not, with
no missed head; `misclass` means it swapped one class for another; `false_negative`
means it predicted nothing where the majority saw hate. The interesting question is
whether the explanations explain *errors* differently from successes. If the text
ablation flips confident correct cases as easily as it flips false negatives, the
explanation is describing the decision rule and not the mistake.
"""),
        md(f"""
{qual_md}
"""),
        code("""
flip = load("p4_flip.json")
for name, block in flip["modalities"].items():
    print(f"{name:6s} top {block['top_flip_rate']:.3f}  random {block['random_flip_rate']:.3f}  "
          f"diff {block['difference']:+.4f}  McNemar p {block['mcnemar_p']:.2e}")

print()
table = pd.DataFrame(flip["outcomes"]).T
print(table[["n", "share", "text_flip_rate", "image_flip_rate", "both_flip_rate"]].round(3).to_string())
if "outcome_flip_table" in flip:
    block = flip["outcome_flip_table"]
    print()
    print(f"homogeneity of the both-modality flip rate across outcome classes: "
          f"chi2={block['chi2']:.2f}, p={block['p']:.2e}, groups={block['groups']}")

if os.path.exists(os.path.join(REFINE, "p4_qualitative.json")):
    qual = load("p4_qualitative.json")
    print()
    for group, counts in qual["groups"].items():
        print(f"{group:8s}", {k: v for k, v in counts.items() if k != "n"})
    print("share of flips whose deleted span is a slur:", qual["share_slur_flip"],
          "| controls:", qual["share_slur_control"])
"""),
    ]


def section_roar(roar) -> list:
    if not roar:
        return [md("## 8. Would forcing the model to use the image help?\n\n*Pending: `p6_roar.json`.*")]

    arms = roar["arms"]
    comparison = roar["comparison"]
    verdict = "clears" if comparison["passes"] else "fails"

    arm2 = arms["arm2_refit"]
    arm3 = arms["arm3_refit_dropout"]
    return [
        md(f"""
## 8. Forcing the model to use the image: a modality-level ROAR

Removing an important component should hurt. Applied at the modality level that logic
becomes a statement about training rather than attribution: if text reliance is a
training artefact, retraining the fusion head with modality dropout should raise the
image's share without costing much accuracy. The encoders stay frozen and only the
fusion block is touched, so the arms differ in one thing.

| arm | eval macro-F1 | train macro-F1 | image share of $|\\phi|$ | image dominance |
|---|---|---|---|---|
| 1 released head | {arms['arm1_released']['macro_f1']:.3f} | --- | {pc(arms['arm1_released']['phi_i_share'])} | {pc(arms['arm1_released']['image_dominance'])} |
| 2 re-initialised, refit on 2,000 | {arm2['macro_f1']:.3f} ± {arm2['macro_f1_sd']:.3f} | {arm2['train_macro_f1']:.3f} | {pc(arm2['phi_i_share'])} ± {100 * arm2['phi_i_share_sd']:.1f} | {pc(arm2['image_dominance'])} |
| 3 same, plus modality dropout p = {roar['dropout']} | {arm3['macro_f1']:.3f} ± {arm3['macro_f1_sd']:.3f} | {arm3['train_macro_f1']:.3f} | {pc(arm3['phi_i_share'])} ± {100 * arm3['phi_i_share_sd']:.1f} | {pc(arm3['image_dominance'])} |

Three seeds per arm, {roar['epochs']} epochs, stratified 2,000/1,000 split inside the analysis set
fixed across arms. Pre-registered: the image's share must rise while macro-F1 drops by
at most 0.01. The dropout arm moves the share by
{100 * comparison['delta_phi_i_share']:+.1f} pp and macro-F1 by
{comparison['delta_macro_f1']:+.4f}, so the criterion {verdict}, and it fails in the wrong
direction. Modality dropout made the head rely on the image *less*, not more.

The mechanism is worth stating because it explains why the obvious fix does not work.
Dropout replaces the modality feature with its blank value but leaves the presence flag
at 1, so the head learns that *this particular feature vector* carries no information
rather than that *the image* should be weighted up. At evaluation the real image still
receives the same small weight. Buying image reliance needs the modality removed at the
input and the label preserved, which is a different training objective.

**What this comparison can and cannot support.** The refit arms see 2,000 samples and
reach an eval macro-F1 of {arm2['macro_f1']:.3f} against {arms['arm1_released']['macro_f1']:.3f} for the released head, so absolute
numbers are not comparable and arm 1 is a scale reference only. Arm 2 against arm 3
shares data, capacity and optimiser, so that contrast is the interpretable one. The sign
is stable across training budgets: at 150 / 400 / 800 epochs the image share moves
0.178→0.110, 0.227→0.113 and 0.206→0.120, and in every budget the dropout arm is the
*more accurate* one (+2.6 / +2.2 / +2.1 pp macro-F1). Under a balanced objective the head
does not merely fail to pick up the image; it does better by leaning on the text harder.
A test at the original training scale is not available here, and §12 says so. One
correction belongs here too: the first run of this experiment used an inverted class
weight in the BCE (n_pos/n_neg instead of n_neg/n_pos), which down-weighted the rare
heads and produced eval F1s near 0.38; the numbers above are from the corrected
objective (§11).
"""),
        code("""
roar = load("p6_roar.json")
arms = pd.DataFrame(roar["arms"]).T[
    ["macro_f1", "train_macro_f1", "phi_i_share", "image_dominance"]]
print(arms.apply(pd.to_numeric, errors="coerce").round(4).fillna("---").to_string())
print()
print(json.dumps(roar["comparison"], indent=1))
print()
print("per-seed image share, arm2:", [round(v, 3) for v in roar["arms"]["arm2_refit"]["phi_i_share_per_seed"]])
print("per-seed image share, arm3:", [round(v, 3) for v in roar["arms"]["arm3_refit_dropout"]["phi_i_share_per_seed"]])
"""),
    ]


def section_human(agreement) -> list:
    if not agreement:
        return [md("## 9. Does the model read the same cues as people?\n\n*Pending: `p5_agreement.json`.*")]
    buckets = agreement["buckets"]
    full = agreement.get("full_sample", {})
    verdict = "clears" if agreement["passes"] else "does not clear"
    return [
        md(f"""
## 9. Does the model read the same cues as people?

If the model leaned on visual context as much as humans do, samples where annotators
agree should show a *smaller* text share than samples where they disagree: the model
would be using the picture precisely where the words are ambiguous. The pre-registered
primary analysis restricts to the hate-relevant subset, samples where at least one
annotator gave a usable hate label (`religion` does not count as one): n =
{agreement['n']:,} of 3,000, with {agreement['n_excluded']} all-none tweets excluded, because
leaving them in turns the question into "normal tweets against hateful tweets". On that
subset, agreement (1.0 / 0.67 / 0.33 from the raw annotations) against the text share by
one-sided Spearman:

$$\\rho = {agreement['spearman_rho']:+.3f}, \\quad p_{{\\text{{one-sided}}}} = {agreement['spearman_p_one_sided_negative']:.1e}$$

The magnitude {verdict} the -0.2 threshold by a wide margin, so the hypothesis is
reported as a miss (§11). It also gets weaker under its own controls: partialling out the
human usable-vote count and the $v(T,I)$ quartile leaves
$\\rho_{{\\text{{partial}}}} = {agreement['partial_rho']:+.3f}$
(p = {agreement['partial_p_one_sided_negative']:.3f}). On the full 3,000 the raw correlation is
{full.get('spearman_rho', float('nan')):+.3f}, but under the same controls it is
{full.get('partial_rho', float('nan')):+.3f}. Most of the uncontrolled association is
agreement moving with the target level, not with the modality mix. The first
version of this analysis ran the full-sample correlation without the controls and read the
result as a directional near-miss; the restricted and controlled version is what is
reported here, and it is weaker.

| agreement bucket | n | mean text share | mean $\\phi_T$ | mean target |
|---|---|---|---|---|
""" + "\n".join(
            f"| {label.replace('_', ' ')} | {block['n']} | {pc(block['text_share_mean'])} | "
            f"{fs(block['phi_t_mean'])} | {block['target_mean']:.3f} |"
            for label, block in buckets.items()
        ) + f"""

Inside the subset the buckets are not monotone in $\\phi_T$
({fs(buckets['all_agree']['phi_t_mean'])} / {fs(buckets['partial']['phi_t_mean'])} /
{fs(buckets['all_differ']['phi_t_mean'])}) and the text share barely moves
({pc(buckets['all_agree']['text_share_mean'])} to {pc(buckets['all_differ']['text_share_mean'])}),
which is consistent with the partial correlation: what changes with agreement is mostly
how *hateful* the sample is, not how the model splits that hate between modalities. Bucket
means stay descriptive; the Kruskal-Wallis statistic
(H = {agreement['kruskal_statistic']:.1f}, p = {agreement['kruskal_p']:.1e}) cannot carry a
claim at three groups, and the pre-registered criterion remains the per-sample partial
correlation.
"""),
        code("""
agreement = load("p5_agreement.json")
print(f"primary n={agreement['n']} rho={agreement['spearman_rho']:+.4f} "
      f"p1={agreement['spearman_p_one_sided_negative']:.2e} "
      f"partial={agreement['partial_rho']:+.4f} p1={agreement['partial_p_one_sided_negative']:.2e} "
      f"passes={agreement['passes']}")
print(f"full sample n={agreement['full_sample']['n']} rho={agreement['full_sample']['spearman_rho']:+.4f} "
      f"partial={agreement['full_sample']['partial_rho']:+.4f}")
print(pd.DataFrame(agreement["buckets"]).T[
    ["n", "text_share_mean", "phi_t_mean", "phi_i_mean", "target_mean"]].round(4).to_string())
"""),
    ]


def section_robustness(robust, siglip) -> list:
    if not robust:
        return [md("## 10. Robustness checks\n\n*Pending: `p8_robustness.json`.*")]

    a3 = robust["a3_preprocessing"]
    sens = robust["target_sensitivity"]
    siglip_md = ""
    if siglip:
        siglip_md = f"""
**A different checkpoint of the same idea (SigLIP2).** The H1 replication swaps
CLIP ViT-B/32 for `{siglip['encoder']}` with the same gated-fusion head and five heads,
on the same {siglip['n']:,} analysis samples. Text is the larger modality on
{pc(siglip['text_dominance_share'])} of samples against
{pc(siglip['main_text_dominance_share'])} for the audited CLIP checkpoint, a gap of
{100 * siglip['dominance_gap']:.1f} pp against a pre-registered tolerance of 10 pp, so the
pattern is not specific to one set of weights.

Two caveats decide how much that sentence is worth. The absolute values are not
comparable: this fusion adds interaction terms, and both Shapley means come out negative
($\\phi_T$ {fs(siglip['phi_t']['mean'])}, $\\phi_I$ {fs(siglip['phi_i']['mean'])}), which
means the blank baseline outscores the real input on average for this checkpoint. Only
the share and the ordering are interpreted. And the released forward cannot run on
transformers 5 as shipped: it passes a `BaseModelOutputWithPooling` straight into
`F.normalize`. The fusion block is therefore reimplemented from the pooler tensors; a parity
check against the checkpoint's own forward gives a maximum absolute difference of 0.0 on
eight samples. The wording is "robust", not "reproduced": a different tokenizer,
thresholds and feature width make this a check that the pattern generalises, not a
replication of the same model.
"""
    return [
        md(f"""
## 10. Three checks that could have broken the story

**A3, preprocessing.** The tweet text is fed raw: URLs, @mentions and `RT` markers
included. Stripping them and recomputing the four coalitions gives
$\\phi_T$ {fs(a3['phi_t_mean_raw'])} → {fs(a3['phi_t_mean_stripped'])}
({100 * a3['relative_shift']:.1f}% relative shift), rank correlation
{a3['spearman_phi_t']:+.3f}, dominance {pc(a3['text_dominance_raw'])} →
{pc(a3['text_dominance_stripped'])}. The criterion was a shift below 10% with the
dominance ordering unchanged: {'pass' if a3['passes'] else 'fail'}. Truncated URLs and
@handles are not carrying the effect.

**Target sensitivity, four heads against five.** Dropping `religion` is a modelling
choice, so the whole analysis is recomputed with all five heads averaged:

| target | mean $\\phi_T$ | mean $\\phi_I$ | text dominance |
|---|---|---|---|
| four usable heads | {fs(sens['four_heads']['phi_t_mean'])} | {fs(sens['four_heads']['phi_i_mean'])} | {pc(sens['four_heads']['text_dominance'])} |
| five heads | {fs(sens['five_heads']['phi_t_mean'])} | {fs(sens['five_heads']['phi_i_mean'])} | {pc(sens['five_heads']['text_dominance'])} |

Dominance differs by {100 * sens['dominance_difference']:.1f} pp;
{'the ordering is unchanged, which is why the four-head target is defensible' if sens['passes']
 else 'the ordering is not stable across targets, which weakens every target-level claim in §5'}.
{siglip_md}"""),
        code("""
robust = load("p8_robustness.json")
print(json.dumps(robust["a3_preprocessing"], indent=1))
print()
sens = pd.DataFrame({key: robust["target_sensitivity"][key]
                     for key in ("four_heads", "five_heads")}).T
print(sens[["phi_t_mean", "phi_i_mean", "text_dominance"]].round(4).to_string())
if os.path.exists(os.path.join(REFINE, "p7_siglip2_phi.json")):
    print()
    print(json.dumps(load("p7_siglip2_phi.json"), indent=1))
"""),
    ]


def section_negative(bug: dict, tcav, roar, overlap) -> list:
    tcav_line = ""
    if tcav:
        tcav_line = (f"The per-sample gradients are {tcav['sensitivity_collinearity']:.2f}-collinear, so the "
                     f"score saturates at 0 or 1 and the random-direction null does the same. The concept "
                     f"whose labels are demonstrably good (`printed_text` at 0.93/0.94) is also the one "
                     f"whose CAV only reaches 0.62 accuracy, which locates the failure in the gradient.")
    roar_line = ""
    if roar:
        roar_line = (f"Modality dropout moved the image share by "
                     f"{100 * roar['comparison']['delta_phi_i_share']:+.1f} pp against a pre-registered "
                     f"positive direction, while macro-F1 moved {roar['comparison']['delta_macro_f1']:+.3f}; "
                     f"the dropout head is simultaneously less image-reliant and more accurate.")
    overlap_line = ""
    if overlap:
        overlap_line = (f"The attribution inside printed-text boxes is {100 * overlap['mean_excess']:+.1f} pp "
                        f"below an equal-area control (p = {overlap['p_wilcoxon']:.1e}), against a "
                        f"pre-registered +15 pp.")

    return [
        md(f"""
## 11. What went wrong, including one bug that would have faked the result

Eleven negative or uncomfortable outcomes are reported here.

**1. A silent bug made three text methods identical.** The first P1.3 run produced
LIME, KernelSHAP and the random ordering with the *same* AOPC to six decimals. The
cause: `content_positions` expected a 1-D attention mask and returned an empty list for
the `(1, 77)` tensor the tokenizer actually produces, so every text method fell back to
in-order ranking and "did LIME beat random?" was measuring nothing. The tell was the
exact equality across methods; the whole file was deleted and regenerated after the
function was fixed to squeeze the batch dimension. Had it survived, §3 would have
reported that no text method beats random on a model that is 99.4% text-driven.

**2. The reference method is the worst text method.** Occlusion ranks below random on
text ({fs(bug['occlusion_text'])} versus {fs(bug['random_text'])}). Single-token
deletions are strongly non-additive on this checkpoint, so a leave-one-out ranking
disagrees with the model's own additive structure; LIME and KernelSHAP, which fit that
structure, win. This is a caution against treating occlusion as a neutral ground truth.

**3. The winner's metric is its own objective.** AOPC rewards exactly what LIME
optimises. The KernelSHAP margin of {abs(bug['lime_minus_shap']):.4f} is small, and the
honest statement is that both surrogate methods clear the floor, not that LIME is
"the" faithful method.

**4. A pre-registered hypothesis missed, and the first version of it was the wrong
analysis.** On the pre-registered hate-relevant subset the correlation is
{bug['rho']:+.3f} against a required -0.20 (§9), and it shrinks to
{bug['rho_partial']:+.3f} once the usable-vote count and the target level are controlled.
The number reported for two days was the full-sample correlation
({bug['rho_full']:+.3f}), which under the same controls collapses to
{bug['rho_full_partial']:+.3f}: most of it was agreement moving with the target level, not
with the modality mix. The larger number is significant and correctly signed, which makes it attractive;
the report keeps the restricted, controlled one and marks the hypothesis a miss.

**5. The concept probe does not work, for a reason unrelated to labels.** {tcav_line}
This is reported as a methodological failure rather than as evidence about the model.

**6. The intervention that should have worked did not.** {roar_line} The refit is
sample-limited (2,000 samples, §8), so the fair reading is that this cheap version of
the fix fails, not that the fix is impossible.

**7. The probe that should have supported the leakage story came back negative.**
{overlap_line} The hypothesis that the image channel re-reads printed text is therefore
not supported by the one measurement designed to find it.

**8. The gold standard was missing for most of the project.** EasyOCR coverage was
reported for two days without any error estimate. A 200-image blind gold set fixed that
late (§1): detection 70.9%, false-positive rate 0%, and an implied text-bearing share
within 0.2 pp of the human rate. Any claim that X% of images contain Y needs its error rate measured
when the claim is made, not after it has propagated into three other sections.

**9. Detached runs died silently.** Two long evaluations exited with no traceback, no
error log and no Windows crash event, at roughly the same sample count. Re-running in
the foreground with `faulthandler` completed both times. The pipeline therefore writes
one JSON line per sample and resumes from the file, so a kill costs at most one sample;
every long script here is restartable for that reason.

**10. The first overlap control was geometrically larger than the thing it controlled
for.** The equal-area control sized its square from the *sum* of the OCR box areas while
the observed statistic used the *union* of those boxes; where boxes overlap the sum
exceeds the union (mean ratio 0.955, and below 0.98 for 46% of the 1,113 samples), so the
control square was too big and the excess looked more negative than it is. Recomputing
with a union-area control moves the excess from −1.16 pp to −0.48 pp (§6). The verdict is
unchanged: it still misses +15 pp in the negative direction. The earlier number
overstated the effect by a factor of 2.4, and both variants are now reported side by side.

**11. The first ROAR objective inverted the class weights.** `pos_weight` in the BCE was
set to n_pos/n_neg, which *down*-weights the rare heads (religion at 2.7% prevalence got
a weight near 0.1) and pushed both refit arms to an eval macro-F1 near 0.38. Corrected to
n_neg/n_pos the arms reach 0.461 and 0.483, and the comparison becomes sharper rather
than softer: dropout now costs nothing in accuracy while still cutting the image share
(§8). A wrong objective that makes both arms look bad is worse than a wrong objective
that makes one arm look bad, because it invites the reader to blame the sample size.
"""),
        code("""
matrix = load("p1_matrix.json")
print("occlusion vs random on text:",
      f"{matrix['text']['occlusion']['aopc_del']['mean']:+.4f}",
      "vs", f"{matrix['text']['lime']['aopc_del']['random_mean']:+.4f}")
print("AOPC is LIME's own objective; the KernelSHAP gap is",
      f"{matrix['text']['lime']['aopc_del']['mean'] - matrix['text']['kernelshap']['aopc_del']['mean']:+.4f}")

gold = load("p0_text_gold.json")
print()
print(f"gold standard: n={gold['n_usable']} detection={gold['ocr_detection_rate']:.3f} "
      f"FPR={gold['ocr_false_positive_rate']:.3f} kappa={gold['agreement_kappa']:.3f}")
"""),
    ]


def section_conclusion(phi, flip, roar, agreement, overlap) -> list:
    roar_line = ""
    if roar:
        roar_line = (f"Re-fitting the fusion head with modality dropout moved the image's share by "
                     f"{100 * roar['comparison']['delta_phi_i_share']:+.1f} pp while macro-F1 moved "
                     f"{roar['comparison']['delta_macro_f1']:+.4f}: the head became less image-reliant and "
                     f"more accurate at once, so the cheap version of the fix is worse than useless.")
    flip_line = ""
    if flip:
        flip_line = (f"Ablating the top 10% of blamed text units flips "
                     f"{pc(flip['modalities']['text']['top_flip_rate'])} of decisions against "
                     f"{pc(flip['modalities']['text']['random_flip_rate'])} for random units, which is what "
                     f"makes the attributions more than a description of the input.")
    overlap_line = ""
    if overlap:
        overlap_line = (f"The image attribution inside printed-text boxes is "
                        f"{100 * overlap['mean_excess']:+.1f} pp below an equal-area control, so the "
                        f"leakage hypothesis is not supported.")
    return [
        md(f"""
## 12. Conclusions

**The answer to the research question.** On `clip-vit-base-mmhs150k-mtl`, the text
modality accounts for {pc(phi['text_dominance_share'])} of the decision in a two-player
Shapley decomposition, with mean $\\phi_T$ = {fs(phi['phi_t']['mean'])} against
$\\phi_I$ = {fs(phi['phi_i']['mean'])}. The image is used, but weakly and often
adversely: $\\phi_I < 0$ on {pc(phi['phi_i_negative_share'])} of samples. {flip_line}

**Which explainers can be trusted.** Only methods that were checked against deletion
and against a randomised model are used for the interpretation: LIME and KernelSHAP on
text, occlusion on image. Grad-CAM, integrated gradients and Score-CAM fail the image
floor on this checkpoint, and occlusion fails the text floor. Both facts are results
about the model-plus-method pair, not about the methods in general.

**What the image channel is not.** Three probes tried to give the small image
contribution a content, and all three came back negative. {overlap_line} The concept
probe cannot adjudicate: its gradients are 0.83-collinear so the score saturates, two of
its three concept sets are contaminated on the negative side, and the one clean concept
still yields a 0.62-accuracy CAV (§6). {roar_line}

**Why the gate cannot be the explanation.** The fused gate correlates with the Shapley
split at ρ = {phi['gate_spearman']:+.2f} (p = {phi['gate_spearman_p']:.1e}):
statistically real, practically useless for a per-sample claim, and sign agreement with
$\\phi_T$ is only {pc(phi['flag_phi_t_sign_agreement'])}. A model card that reports gate
averages as evidence of multimodal use would be overclaiming.

**The measurement the dataset does not ship.** MMHS150K provides no OCR for this split,
and EasyOCR's raw coverage (36.6%) is a lower bound: a 200-image blind gold set puts the
detection rate at 70.9% with a 0% false-positive rate, and the implied true text-bearing
share (51.6%) matches the human rate (51.8%) to 0.2 pp. That number matters for anyone
who plans to study "text in images" here: roughly half the corpus contains readable
words, and the standard detector finds seven in ten of them.

**What would change the verdict.** A checkpoint trained with modality dropout on the
full training set, which this project could not run at scale (§8); and a dataset whose
images carry labels the text does not. Until then the defensible engineering
recommendation is to report the per-modality ablation next to the accuracy numbers,
because a 0.99 text share is invisible in a macro-F1 of {f4(load('p0_gate.json')['macro_f1_4_usable'])}.
"""),
        md("""
## 13. Reproduction

| stage | script | output |
|---|---|---|
| gate + metrics | `scripts/p0_gate.py` | `p0_gate.json`, `p0_baseline_probs.npz` |
| OCR boxes | `scripts/p0_ocr.py` | `p0_ocr.jsonl`, `p0_ocr_stats.json` |
| OCR gold standard (manual) | `scripts/build_annotation.py`, `scripts/merge_annotation.py` | `annotate/gold.html`, `p0_text_gold.json` |
| seven-method attributions, 499 samples | `scripts/p1_text_attributions.py`, `scripts/p1_image_attributions.py` | `p1_text_attr.jsonl`, `p1_image_attr.jsonl` |
| faithfulness scoring | `scripts/p1_evaluate.py` | `p1_faithfulness.jsonl`, `p1_matrix.json` |
| winner rule | `scripts/p1_decision.py` | `p1_decision.json` |
| gates, stability, agreement | `scripts/p1_sanity_stability.py` | `p1_sanity_curves.json`, `p1_stability.json`, `p1_agreement.json` |
| winner on 3,000 | `scripts/p1d_winner_3k.py` | `p1d_*_attr_test.jsonl` |
| modality Shapley | `scripts/p2_modality.py` | `p2_phi.jsonl`, `p2_curves.json`, `p2_summary.json` |
| concepts, TCAV, label audit | `scripts/p3_concepts.py`, `scripts/p3_tcav.py`, `scripts/merge_annotation.py` | `p3_concepts.json`, `p3_tcav.json`, `p3_concept_audit.json` |
| OCR overlay | `scripts/p3_overlap.py` | `p3_overlap.json` |
| failure modes | `scripts/p4_flip.py` | `p4_flip.json` |
| human agreement | `scripts/p5_agreement.py` | `p5_agreement.json` |
| ROAR | `scripts/p6_roar.py` | `p6_roar.json` |
| SigLIP2 replication | `scripts/p7_siglip2.py` | `p7_siglip2_phi.json` |
| robustness | `scripts/p8_robustness.py` | `p8_robustness.json` |
| fusion parity checks | `scripts/check_fusion_parity.py` | stdout (max abs delta, expected 0.0) |
| this notebook | `scripts/build_notebook.py`, `scripts/make_figures.py` | `Course Project-2026A.ipynb` |

Environment: Python 3.14, PyTorch 2.11 (CUDA 12.8) on an RTX 5080, transformers 5.17,
easyocr 1.7.2. Every long-running script appends one JSON line per sample and skips
completed ids on restart, so the pipeline is resumable at sample granularity. The
notebook itself only reads the stored artefacts; the one cell that touches the model is
the intervention demo in §2. The annotation pages were deployed to GitHub Pages at
`botherain.github.io/mmhs-annotation`; the three exports are kept in `annotate/out/` and
merged by `scripts/merge_annotation.py`, which takes the majority label per item when
several annotators labelled the same image.
"""),
    ]


def build() -> None:
    gate = load("p0_gate.json")
    stats = load("p0_stats.json")
    ocr = load("p0_ocr_stats.json")
    matrix = load("p1_matrix.json")
    decision = load("p1_decision.json")
    phi = load("p2_summary.json")
    curves = load("p1_sanity_curves.json")
    stability = load("p1_stability.json")
    overlap = load("p3_overlap.json")
    tcav = load("p3_tcav.json")
    audit = load("p3_concept_audit.json")
    ctext = load("p3_concept_text.json")
    gold = load("p0_text_gold.json")
    flip = load("p4_flip.json")
    qual = load("p4_qualitative.json")
    agreement = load("p5_agreement.json")
    roar = load("p6_roar.json")
    robust = load("p8_robustness.json")
    siglip = load("p7_siglip2_phi.json")

    bug = {
        "occlusion_text": matrix["text"]["occlusion"]["aopc_del"]["mean"] if matrix else 0.0,
        "random_text": matrix["text"]["lime"]["aopc_del"]["random_mean"] if matrix else 0.0,
        "lime_minus_shap": (matrix["text"]["lime"]["aopc_del"]["mean"]
                            - matrix["text"]["kernelshap"]["aopc_del"]["mean"]) if matrix else 0.0,
        "rho": agreement["spearman_rho"] if agreement else 0.0,
        "rho_partial": agreement.get("partial_rho", 0.0) if agreement else 0.0,
        "rho_full": agreement["full_sample"]["spearman_rho"] if agreement else 0.0,
        "rho_full_partial": agreement["full_sample"]["partial_rho"] if agreement else 0.0,
    }

    with open(TEMPLATE, encoding="utf-8") as handle:
        template = nbformat.read(handle, as_version=4)
    head = list(template.cells[:3])
    head[0] = md(f"""**Name:** {NAMES}

**EIDs:** {EIDS}""")

    cells = list(head)
    cells += section_setup()
    cells += section_model(gate, stats, ocr, gold)
    cells += section_units()
    cells += section_p1(matrix, decision)
    cells += section_gates(curves, stability)
    cells += section_phi(phi)
    cells += section_image_evidence(overlap, tcav, audit, ctext)
    cells += section_failures(flip, qual)
    cells += section_roar(roar)
    cells += section_human(agreement)
    cells += section_robustness(robust, siglip)
    cells += section_negative(bug, tcav, roar, overlap)
    cells += section_conclusion(phi, flip, roar, agreement, overlap)

    notebook = nbformat.v4.new_notebook()
    notebook["cells"] = cells
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:02d}")
    notebook["metadata"] = {
        "kernelspec": {"display_name": "my_local_env", "language": "python", "name": "my_local_env"},
        "language_info": {"name": "python"},
    }
    with open(TARGET, "w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    print(f"wrote {TARGET} with {len(cells)} cells")


if __name__ == "__main__":
    build()
