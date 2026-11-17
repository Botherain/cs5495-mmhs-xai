"""Build the P4.2 qualitative review page: 20 text-ablation flips + 20 controls.

The stats in §7 say the top-10% text ablation flips 38.8% of decisions. They do not say
*what* was removed when it flipped, which is the question a reader actually asks. This
page shows 40 cases --- 20 flips and 20 matched controls --- with the deleted tokens
highlighted, the before/after head probabilities, and the human majority label, so a
human can code what the ablation removed and whether the flip is meaningful.

Everything is reconstructed from stored artefacts (p4_flip.jsonl for the probabilities,
p1d_text_attr_test.jsonl for the LIME ranking, the tokenizer for the token strings), so
building the page costs no model time.

Output: annotate/qualitative.html

Usage: python scripts/build_qualitative.py
"""

from __future__ import annotations

import json
import os
import random

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "refine-logs")
ANNOTATE = os.path.join(ROOT, "annotate")
DATA = os.path.join(ROOT, "data", "MMHS150K")
SEED = 20261114

HEADS = ["racist", "sexist", "homophobe", "religion", "otherhate"]
THRESHOLDS = [0.3, 0.7, 0.5, 0.3, 0.55]
K_FRACTION = 0.1

# 20 flips and 20 controls, matched on the error class so the page is not all
# false positives.
QUOTA = {"false_positive": 8, "correct": 7, "over_trigger": 4, "misclass": 1}


def load_jsonl(name: str) -> dict:
    rows = {}
    with open(os.path.join(OUT, name), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                rows[record["id"]] = record
    return rows


def pick(pool: list, n: int, rng: random.Random) -> list:
    rng.shuffle(pool)
    return pool[:n]


def main() -> None:
    from transformers import CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    with open(os.path.join(DATA, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    flip = load_jsonl("p4_flip.jsonl")
    attributions = load_jsonl("p1d_text_attr_test.jsonl")

    flips, controls = {}, {}
    for tweet_id, record in flip.items():
        bucket = flips if record["text_flip"] else controls
        bucket.setdefault(record["outcome"], []).append(tweet_id)

    rng = random.Random(SEED)
    chosen = []
    for outcome, quota in QUOTA.items():
        chosen += [(tweet_id, True) for tweet_id in pick(list(flips.get(outcome, [])), quota, rng)]
        chosen += [(tweet_id, False) for tweet_id in pick(list(controls.get(outcome, [])), quota, rng)]
    rng.shuffle(chosen)

    cases = []
    for tweet_id, is_flip in chosen:
        record = flip[tweet_id]
        attribution = attributions[tweet_id]
        positions = [int(position) for position in attribution["positions"]]
        values = np.asarray(attribution["values"], dtype=float)
        k = int(max(1, round(len(positions) * K_FRACTION)))
        dropped = {positions[index] for index in np.argsort(-values)[:k]}

        text = gt[tweet_id].get("tweet_text", "")
        ids = tokenizer(text, truncation=True, max_length=77)["input_ids"]
        tokens = []
        for position in positions:
            if position >= len(ids):
                continue
            piece = tokenizer.decode([ids[position]]).strip()
            if piece:
                tokens.append({"t": piece, "hit": position in dropped})

        cases.append({
            "id": tweet_id,
            "img": f"../data/MMHS150K/img_resized/{tweet_id}.jpg",
            "text": text,
            "tokens": tokens,
            "n_dropped": len(dropped),
            "base": [round(float(v), 4) for v in record["base"]],
            "after": [round(float(v), 4) for v in record["text_probs"]],
            "base_set": record["base_set"],
            "truth_set": record["truth_set"],
            "outcome": record["outcome"],
            "flip": bool(is_flip),
        })

    payload = {
        "heads": HEADS,
        "thresholds": THRESHOLDS,
        "seed": SEED,
        "cases": cases,
    }
    html = PAGE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    with open(os.path.join(ANNOTATE, "qualitative.html"), "w", encoding="utf-8") as handle:
        handle.write(html)

    n_flip = sum(1 for case in cases if case["flip"])
    print(f"cases {len(cases)} ({n_flip} flips / {len(cases) - n_flip} controls) -> annotate/qualitative.html")
    for case in cases[:3]:
        print(" ", case["outcome"], case["flip"], [t["t"] for t in case["tokens"] if t["hit"]])


PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>P4.2 定性检视 · 40 条</title>
<style>
  body { margin: 0; background: #14171c; color: #e8eaed; font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
  header { position: sticky; top: 0; z-index: 10; background: #1d2229; border-bottom: 1px solid #2b323c;
           padding: 10px 18px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .bar { flex: 1; min-width: 160px; height: 8px; background: #2b323c; border-radius: 4px; overflow: hidden; }
  .bar > i { display: block; height: 100%; width: 0; background: #3f8cff; }
  .stat { font-size: 13px; color: #9aa4b2; font-variant-numeric: tabular-nums; }
  main { max-width: 980px; margin: 0 auto; padding: 18px; display: flex; flex-direction: column; gap: 18px; }
  .card { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 16px; background: #1a1f26;
          border: 1px solid #262d36; border-radius: 10px; padding: 14px; }
  .card.done { border-color: #2f5f45; }
  .card img { width: 240px; border-radius: 6px; background: #0d1014; }
  .meta { font-size: 12px; color: #9aa4b2; margin-bottom: 6px; }
  .meta b { color: #cfd6df; font-weight: 600; }
  .raw { font-size: 13px; color: #c2cad4; background: #14171c; border: 1px solid #262d36;
         border-radius: 6px; padding: 8px; margin: 8px 0; white-space: pre-wrap; word-break: break-word; }
  .tokens { font-family: Consolas, monospace; font-size: 12.5px; line-height: 2.0; margin: 8px 0; }
  .tok { padding: 1px 4px; margin: 1px; border-radius: 4px; background: #222933; display: inline-block; }
  .tok.hit { background: #7a2b2b; color: #ffd9d9; text-decoration: line-through; }
  table.probs { border-collapse: collapse; font-size: 12px; margin: 6px 0 10px; }
  table.probs th, table.probs td { padding: 3px 8px; border-bottom: 1px solid #262d36; text-align: right; }
  table.probs th:first-child, table.probs td:first-child { text-align: left; color: #9aa4b2; }
  .q { font-size: 13px; margin-top: 10px; }
  .q > div:first-child { color: #cfd6df; margin-bottom: 5px; }
  label.chk { display: inline-block; font-size: 12.5px; margin: 0 10px 4px 0; color: #c2cad4; cursor: pointer; }
  label.rad { display: block; font-size: 12.5px; margin: 2px 0; color: #c2cad4; cursor: pointer; }
  textarea { width: 100%; box-sizing: border-box; margin-top: 8px; min-height: 44px; background: #14171c;
             color: #e8eaed; border: 1px solid #333c48; border-radius: 6px; padding: 6px; font: inherit; font-size: 12.5px; }
  input[type=text] { background: #222933; border: 1px solid #333c48; color: #e8eaed; border-radius: 6px;
                     padding: 5px 8px; font: inherit; font-size: 12px; width: 110px; }
  button { font: inherit; font-size: 12px; border-radius: 8px; border: 1px solid #333c48; background: #222933;
           color: #e8eaed; padding: 8px 12px; cursor: pointer; }
  button:hover { background: #2a3340; }
  .hint { font-size: 12px; color: #8b95a3; }
</style>
</head>
<body>
<header>
  <h1>P4.2 定性检视 · 20 翻转 + 20 对照</h1>
  <input id="who" placeholder="标注者代号">
  <div class="bar"><i id="bar"></i></div>
  <div class="stat" id="stat"></div>
  <button id="export">导出结果</button>
</header>
<main id="main"></main>
<div class="hint" style="max-width:980px;margin:0 auto;padding:0 18px 24px">
  每条是一次干预：把归因排序前 10% 的 token 从 attention mask 里去掉，看决策是否改变。
  文字条带是 BPE 片段，划掉的是被删部分；上方 raw 行是完整推文。前 20 条与后 20 条按错误类型配对，标记为「翻转 / 对照」。
</div>
<script>
const DATA = __DATA__;
const STORE_KEY = "mmhs_qualitative";
const WHO_KEY = "mmhs_who";
let store = {};
try { store = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { store = {}; }
let storageOk = true;

const who = document.getElementById("who");
try { who.value = localStorage.getItem(WHO_KEY) || ""; } catch (e) {}
who.onchange = () => { try { localStorage.setItem(WHO_KEY, who.value.trim()); } catch (e) {} };

const TAGS = [
  ["slur", "侮辱词 / 群体指称"],
  ["negation", "否定 / 反讽 / 程度修饰"],
  ["link", "URL / @提及 / RT"],
  ["content", "普通实词"],
  ["function", "停用词 / 标点 / 表情"],
];
const VERDICTS = [
  ["justified", "翻转合理：删掉的确实是决定性的内容"],
  ["artefact", "翻转是伪影：删掉的东西与仇恨判断无关"],
  ["unclear", "说不清"],
];

function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); } catch (e) { storageOk = false; }
}

function card(item, index) {
  const entry = store[item.id] || {};
  const el = document.createElement("div");
  el.className = "card" + (entry.verdict ? " done" : "");
  el.id = "case-" + item.id;

  const heads = DATA.heads.map((name, i) => {
    const base = item.base[i], after = item.after[i], thr = DATA.thresholds[i];
    const mark = (v) => v > thr ? "<b style='color:#f2a2a2'>" + v.toFixed(3) + "</b>" : v.toFixed(3);
    return `<tr><td>${name}</td><td>${mark(base)}</td><td>${mark(after)}</td><td>${thr}</td></tr>`;
  }).join("");

  const tokens = item.tokens.map(t =>
    `<span class="tok${t.hit ? " hit" : ""}">${t.t.replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</span>`
  ).join("");

  const truth = item.truth_set.length ? item.truth_set.map(i => DATA.heads[i]).join(", ") : "(none)";
  const base = item.base_set.length ? item.base_set.map(i => DATA.heads[i]).join(", ") : "(none)";

  el.innerHTML = `
    <div><img src="${item.img}" alt=""></div>
    <div>
      <div class="meta">#${index + 1} · <b>${item.flip ? "翻转" : "对照（未翻转）"}</b> · 错误类型 ${item.outcome} · 删掉 ${item.n_dropped} 个 token</div>
      <div class="raw">${item.text.replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</div>
      <div class="tokens">${tokens}</div>
      <table class="probs">
        <tr><th>head</th><th>原输入</th><th>删掉 top-10% 后</th><th>阈值</th></tr>
        ${heads}
      </table>
      <div class="meta">人类多数集合：<b>${truth}</b> · 模型原预测：<b>${base}</b></div>

      <div class="q"><div>1 · 被删掉的 token 属于哪类（可多选）</div>
        ${TAGS.map(([v, label]) => `<label class="chk"><input type="checkbox" data-id="${item.id}" data-tag="${v}" ${entry.tags && entry.tags.includes(v) ? "checked" : ""}> ${label}</label>`).join("")}
      </div>
      <div class="q"><div>2 · 这次变化说明什么</div>
        ${VERDICTS.map(([v, label]) => `<label class="rad"><input type="radio" name="v-${item.id}" data-id="${item.id}" data-verdict="${v}" ${entry.verdict === v ? "checked" : ""}> ${label}</label>`).join("")}
      </div>
      <textarea data-id="${item.id}" placeholder="备注（可选）：删掉的是什么词、为什么这个翻转合理或不合理">${entry.note || ""}</textarea>
    </div>`;
  return el;
}

function render() {
  const main = document.getElementById("main");
  main.innerHTML = "";
  DATA.cases.forEach((item, index) => main.appendChild(card(item, index)));
  const done = DATA.cases.filter(item => store[item.id] && store[item.id].verdict).length;
  document.getElementById("bar").style.width = (100 * done / DATA.cases.length).toFixed(1) + "%";
  document.getElementById("stat").textContent = `${done} / ${DATA.cases.length}` + (storageOk ? "" : " · 本地存储不可用，请常导出");
}

document.addEventListener("change", (event) => {
  const id = event.target.dataset.id;
  if (!id) return;
  store[id] = store[id] || { tags: [], verdict: "", note: "" };
  if (event.target.dataset.tag) {
    const tag = event.target.dataset.tag;
    const tagset = new Set(store[id].tags || []);
    event.target.checked ? tagset.add(tag) : tagset.delete(tag);
    store[id].tags = [...tagset];
  }
  if (event.target.dataset.verdict) store[id].verdict = event.target.dataset.verdict;
  save();
  const cardEl = document.getElementById("case-" + id);
  if (cardEl) cardEl.className = "card" + (store[id].verdict ? " done" : "");
  const done = DATA.cases.filter(item => store[item.id] && store[item.id].verdict).length;
  document.getElementById("bar").style.width = (100 * done / DATA.cases.length).toFixed(1) + "%";
  document.getElementById("stat").textContent = `${done} / ${DATA.cases.length}`;
});

document.addEventListener("input", (event) => {
  const id = event.target.dataset.id;
  if (!id || event.target.tagName !== "TEXTAREA") return;
  store[id] = store[id] || { tags: [], verdict: "", note: "" };
  store[id].note = event.target.value;
  save();
});

document.getElementById("export").onclick = () => {
  const rows = DATA.cases.map(item => ({
    id: item.id,
    outcome: item.outcome,
    flip: item.flip,
    tags: (store[item.id] || {}).tags || [],
    verdict: (store[item.id] || {}).verdict || "",
    note: (store[item.id] || {}).note || "",
    annotator: who.value.trim() || "anon",
  }));
  const blob = new Blob([JSON.stringify({ rows, exported: new Date().toISOString() }, null, 1)],
                        { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "qualitative_" + (who.value.trim() || "anon").replace(/[^\w.-]/g, "_") + "_labels.json";
  link.click();
};

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
