"""Build the two blind annotation pages for the manual labelling steps.

  annotate/gold.html      200 random analysis images -> OCR gold standard (P0.3/P3.4)
  annotate/concept.html   1,824 concept-set images -> CAV training QC (P0.5)

Both are single self-contained HTML files: the manifest is embedded (file:// pages
cannot fetch JSON), the label store is localStorage, and export writes a JSON file
that scripts/merge_annotation.py consumes.

The pages are blind on purpose: no OCR output, no concept name, no polarity. A
label that has seen the detector output is not a gold standard, and a label that
has seen "this is a printed_text positive" measures suggestibility.

Usage: python scripts/build_annotation.py
"""

from __future__ import annotations

import json
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "refine-logs")
ANNOTATE = os.path.join(ROOT, "annotate")
IMG = os.path.join(ROOT, "data", "MMHS150K", "img_resized")
SEED = 20261113
N_GOLD = 200
N_DOUBLE = 100
N_MEMBERSHIP = 100


def load_json(name: str):
    with open(os.path.join(OUT, name), encoding="utf-8") as handle:
        return json.load(handle)


PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
         background: #14171c; color: #e8eaed; }
  header { display: flex; align-items: center; gap: 16px; padding: 10px 16px;
            background: #1d2229; border-bottom: 1px solid #2b323c; flex-wrap: wrap; }
  header h1 { font-size: 15px; font-weight: 600; margin: 0 8px 0 0; }
  .bar { position: relative; flex: 1; min-width: 180px; height: 8px; background: #2b323c;
          border-radius: 4px; overflow: hidden; }
  .bar > i { position: absolute; inset: 0 auto 0 0; width: 0; background: #3f8cff; }
  .stat { font-variant-numeric: tabular-nums; font-size: 13px; color: #9aa4b2; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px;
          padding: 16px; align-items: start; }
  .stage { background: #0d1014; border: 1px solid #262d36; border-radius: 8px;
            min-height: 70vh; display: flex; align-items: center; justify-content: center;
            padding: 10px; }
  .stage img { max-width: 100%; max-height: 74vh; border-radius: 4px; }
  aside { display: flex; flex-direction: column; gap: 10px; }
  button { font: inherit; border-radius: 8px; border: 1px solid #333c48; background: #222933;
            color: #e8eaed; padding: 12px 14px; cursor: pointer; text-align: left; }
  button:hover { background: #2a3340; }
  button:disabled { opacity: .35; cursor: default; }
  button b { display: block; font-size: 15px; margin-bottom: 2px; }
  button span { font-size: 12px; color: #9aa4b2; }
  .key { float: right; font-size: 11px; padding: 1px 6px; border-radius: 4px;
          background: #333c48; color: #cfd6df; }
  .yes b { color: #6ee7a8; } .no b { color: #f2a2a2; } .unsure b { color: #f5cf7a; }
  .tools { display: flex; gap: 8px; flex-wrap: wrap; }
  .tools button { flex: 1; padding: 8px; font-size: 12px; text-align: center; }
  .hint { font-size: 12px; color: #8b95a3; line-height: 1.5; }
  .prompt { font-size: 14px; line-height: 1.5; background: #1d2229; border-radius: 8px;
            border: 1px solid #2b323c; padding: 10px 12px; }
  .prompt b { color: #8fb8ff; }
  .done { padding: 24px; text-align: center; color: #cfd6df; }
  a { color: #7fb3ff; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <input id="who" placeholder="标注者代号" title="导出的文件名会带这个名字"
         style="width:110px;background:#222933;border:1px solid #333c48;color:#e8eaed;
                border-radius:6px;padding:5px 8px;font:inherit;font-size:12px">
  <div class="bar"><i id="bar"></i></div>
  <div class="stat" id="stat"></div>
</header>
<main>
  <div class="stage" id="stage"></div>
  <aside>
    <div class="prompt" id="prompt"></div>
    <button class="yes" id="b1">__B1__</button>
    <button class="no" id="b2">__B2__</button>
    <button class="unsure" id="b3">__B3__</button>
    <div class="tools">
      <button id="back">← 上一张</button>
      <button id="skip">跳过</button>
    </div>
    <div class="tools">
      <button id="export">导出结果</button>
      <button id="import">导入备份</button>
    </div>
    <input type="file" id="file" accept="application/json" hidden>
    <div class="hint" id="hint"></div>
  </aside>
</main>
<script>
const MANIFEST = __MANIFEST__;
const STORE_KEY = "mmhs_has_text::__STORE__";
const TITLE = __TITLE_JSON__;
const V1 = __V1__, V2 = __V2__, V3 = __V3__;
const WHO_KEY = "mmhs_who";

let store = {};
let storageOk = true;
try {
  store = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
} catch (error) {
  storageOk = false;
}

const whoInput = document.getElementById("who");
try { whoInput.value = localStorage.getItem(WHO_KEY) || ""; } catch (error) { /* ignore */ }
whoInput.onchange = () => {
  try { localStorage.setItem(WHO_KEY, whoInput.value.trim()); } catch (error) { /* ignore */ }
};
const who = () => (whoInput.value.trim() || "anon").replace(/[^\w.-]/g, "_");
let index = 0;
let history = [];

const stage = document.getElementById("stage");
const bar = document.getElementById("bar");
const stat = document.getElementById("stat");
const hint = document.getElementById("hint");

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(store));
  } catch (error) {
    storageOk = false;
  }
}

// The 100 blind repeats share an id with their first pass, so the store has to
// be keyed by (id, pass) --- keying by id alone would silently overwrite pass 1
// and make the agreement figure a tautology.
function keyOf(item) { return item.id + "::" + item.pass + "::" + (item.concept || ""); }

function firstUnlabelled() {
  for (let i = 0; i < MANIFEST.length; i++) {
    if (!(keyOf(MANIFEST[i]) in store)) return i;
  }
  return MANIFEST.length;
}

function render() {
  const done = MANIFEST.filter((item) => keyOf(item) in store).length;
  bar.style.width = (100 * done / MANIFEST.length).toFixed(1) + "%";
  stat.textContent = `${done} / ${MANIFEST.length}`;

  if (index >= MANIFEST.length) {
    stage.innerHTML = '<div class="done"><p><strong>这一页标完了。</strong></p>' +
      '<p>点“导出结果”，把下载的 JSON 放回项目的 annotate/out/ 目录。</p></div>';
    document.getElementById("prompt").style.display = "none";
    hint.textContent = "";
    return;
  }

  const item = MANIFEST[index];
  stage.innerHTML = `<img src="${item.img}" alt="">`;
  const promptBox = document.getElementById("prompt");
  if (item.prompt) {
    promptBox.innerHTML = item.prompt;
    promptBox.style.display = "block";
  } else {
    promptBox.style.display = "none";
  }
  const label = store[keyOf(item)];
  hint.innerHTML = (label ? `本张已标：${label.label}（可改）<br>` : "") +
    `第 ${index + 1} 张 · 剩余 ${MANIFEST.length - done}` +
    (storageOk ? "" : '<br><strong style="color:#f5cf7a">本地存储被禁用，请每 50 张导出一次。</strong>');
}

function setLabel(value) {
  if (index >= MANIFEST.length) return;
  const item = MANIFEST[index];
  store[keyOf(item)] = { label: value, queue: item.queue, pass: item.pass, ts: Date.now() };
  save();
  history.push(index);
  index = index + 1 < MANIFEST.length ? index + 1 : MANIFEST.length;
  render();
}

function step(delta) {
  if (delta < 0 && history.length) {
    index = history.pop();
  } else {
    index = Math.max(0, Math.min(MANIFEST.length - 1, index + delta));
  }
  render();
}

document.getElementById("b1").onclick = () => setLabel(V1);
document.getElementById("b2").onclick = () => setLabel(V2);
document.getElementById("b3").onclick = () => setLabel(V3);
document.getElementById("back").onclick = () => step(-1);
document.getElementById("skip").onclick = () => step(1);

document.getElementById("export").onclick = () => {
  const payload = {
    store: "__STORE__",
    title: TITLE,
    annotator: who(),
    exported: new Date().toISOString(),
    rows: MANIFEST.filter((item) => keyOf(item) in store).map((item) => ({
      id: item.id, queue: item.queue, pass: item.pass, concept: item.concept || "",
      side: item.side || "", label: store[keyOf(item)].label,
      annotator: who(),
    })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "__STORE___" + who() + "_labels.json";
  link.click();
};

document.getElementById("import").onclick = () => document.getElementById("file").click();document.getElementById("file").onchange = (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const payload = JSON.parse(reader.result);
      (payload.rows || []).forEach((row) => {
        store[row.id] = { label: row.label, queue: row.queue, pass: row.pass, ts: Date.now() };
      });
      save();
      render();
    } catch (error) {
      alert("导入失败：" + error.message);
    }
  };
  reader.readAsText(file);
};

document.addEventListener("keydown", (event) => {
  if (event.key === "1" || event.key === "ArrowRight") { event.preventDefault(); setLabel(V1); }
  else if (event.key === "2") { event.preventDefault(); setLabel(V2); }
  else if (event.key === "3") { event.preventDefault(); setLabel(V3); }
  else if (event.key === "ArrowLeft") { event.preventDefault(); step(-1); }
});

index = firstUnlabelled();
render();
</script>
</body>
</html>
"""


def build_page(title: str, store: str, items: list, labels: list) -> str:
    manifest = [{"id": item["id"], "img": item["img"], "queue": item["queue"],
                 "pass": item["pass"], "prompt": item.get("prompt", ""),
                 "concept": item.get("concept", ""), "side": item.get("side", "")}
                for item in items]
    html = (PAGE
            .replace("__TITLE_JSON__", json.dumps(title, ensure_ascii=False))
            .replace("__TITLE__", title)
            .replace("__STORE__", store)
            .replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False)))
    for index, (value, button, note) in enumerate(labels, start=1):
        html = html.replace(f"__V{index}__", json.dumps(value, ensure_ascii=False))
        html = html.replace(
            f"__B{index}__",
            f'<b>{index} · {button}</b><span>{note}<span class="key">{index}</span></span>',
        )
    return html


def rel_path(tweet_id: str, base_url: str = "") -> str:
    if base_url:
        return f"{base_url.rstrip('/')}/img/{tweet_id}.jpg"
    return f"../data/MMHS150K/img_resized/{tweet_id}.jpg"


def publish_images(directory: str, tweet_ids: list) -> int:
    """Copy only the images the pages reference, flat, under <directory>/img/."""
    import shutil

    target = os.path.join(directory, "img")
    os.makedirs(target, exist_ok=True)
    copied = 0
    for tweet_id in sorted(set(tweet_ids)):
        source = os.path.join(IMG, f"{tweet_id}.jpg")
        destination = os.path.join(target, f"{tweet_id}.jpg")
        if os.path.exists(source) and not os.path.exists(destination):
            shutil.copy2(source, destination)
            copied += 1
    return copied


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="",
                        help="absolute prefix for image URLs, e.g. https://name.pages.dev")
    parser.add_argument("--publish", default="",
                        help="also copy the referenced images into <dir>/img/ for upload")
    args = parser.parse_args()
    base = args.base_url

    os.makedirs(ANNOTATE, exist_ok=True)
    with open(os.path.join(ROOT, "data", "MMHS150K", "subset_ids.json"), encoding="utf-8") as handle:
        subset = json.load(handle)
    concepts = load_json("p3_concepts.json")["concepts"]

    text_labels = [
        ("text", "有印刷文字", "图里能读出字母或汉字（招牌、截图、字幕、水印都算）"),
        ("none", "没有", "只有照片/插画/图形，没有任何可读文字"),
        ("unclear", "看不清", "太小、太糊、被遮挡，无法判断"),
    ]
    member_labels = [
        ("yes", "是", "这张图确实属于上面那个概念"),
        ("no", "不是", "这张图不属于上面那个概念"),
        ("unclear", "不确定", "太小、太糊，或处在概念边界"),
    ]

    rng = random.Random(SEED)
    gold_ids = rng.sample(list(subset["main"]), N_GOLD)
    gold = [{"id": tweet_id, "img": rel_path(tweet_id, base), "queue": "gold", "pass": 1}
            for tweet_id in gold_ids if os.path.exists(os.path.join(IMG, f"{tweet_id}.jpg"))]

    unique = sorted({tweet_id for block in concepts.values()
                     for side in ("positive", "negative") for tweet_id in block[side]})
    unique = [tweet_id for tweet_id in unique if os.path.exists(os.path.join(IMG, f"{tweet_id}.jpg"))]
    rng.shuffle(unique)
    concept = [{"id": tweet_id, "img": rel_path(tweet_id, base), "queue": "concept", "pass": 1}
               for tweet_id in unique]
    # 100 repeats at the end: the same judgement made twice, blindly, for the
    # inter-rater agreement the plan asks for (>= 0.8).
    repeat = rng.sample(unique, min(N_DOUBLE, len(unique)))
    concept += [{"id": tweet_id, "img": rel_path(tweet_id, base), "queue": "concept", "pass": 2}
                for tweet_id in repeat]

    # Concept-membership audit: the concept set was built from detector cues, not
    # from a human pass, so the only way to know how good those labels are is to
    # label a stratified sample and report each cue's precision and recall.
    prompts = {
        "printed_text": "这张图里<b>有没有可读的文字</b>？招牌、字幕、截图里的字、水印、贴纸字、"
                        "条形码和孤立数字都算；纯照片和图形不算。",
        "face": "这张图里<b>有没有人脸</b>？能看清五官的正面或接近正面人脸算；"
                "侧脸、背影、只有眼睛不算。",
        "flat_graphic": "这张图<b>是不是平涂 / 界面 / 文字块类图形</b>？大面积纯色、截图、表情包、"
                        "海报、logo 算；有明显自然照片细节的实景不算。",
    }
    membership = []
    for name, block in concepts.items():
        for side in ("positive", "negative"):
            pool = list(block[side])
            sample = rng.sample(pool, min(N_MEMBERSHIP, len(pool)))
            membership += [{"id": tweet_id, "img": rel_path(tweet_id, base), "queue": "membership",
                            "pass": 1, "prompt": prompts[name], "concept": name, "side": side}
                           for tweet_id in sample]
    rng.shuffle(membership)

    with open(os.path.join(ANNOTATE, "gold.html"), "w", encoding="utf-8") as handle:
        handle.write(build_page("OCR 金标准 · 200 张", "gold", gold, text_labels))
    with open(os.path.join(ANNOTATE, "concept.html"), "w", encoding="utf-8") as handle:
        handle.write(build_page("概念集 QC · 图内有无印刷文字", "concept", concept, text_labels))
    with open(os.path.join(ANNOTATE, "membership.html"), "w", encoding="utf-8") as handle:
        handle.write(build_page("概念归属抽检 · 600 张", "membership", membership, member_labels))

    manifest = {
        "gold": {"n": len(gold), "title": "OCR gold standard", "file": "gold.html"},
        "concept": {"n_primary": len(unique), "n_double": len(repeat), "total": len(concept),
                    "title": "concept-set QC", "file": "concept.html"},
        "membership": {"n": len(membership), "file": "membership.html",
                       "note": "stratified 100 positive + 100 negative per concept"},
        "seed": SEED,
        "gold_ids": gold_ids,
        "concept_ids": unique,
        "double_ids": repeat,
        "membership_items": [{"id": item["id"], "concept": item["concept"], "side": item["side"]}
                             for item in membership],
        "labels": {"text_labels": [item[0] for item in text_labels],
                   "membership_labels": [item[0] for item in member_labels]},
    }
    with open(os.path.join(ANNOTATE, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.makedirs(os.path.join(ANNOTATE, "out"), exist_ok=True)
    print(f"gold: {len(gold)} images -> annotate/gold.html")
    print(f"concept: {len(unique)} primary + {len(repeat)} double = {len(concept)} -> annotate/concept.html")
    print(f"membership: {len(membership)} images -> annotate/membership.html")

    if args.publish:
        import shutil

        os.makedirs(args.publish, exist_ok=True)
        for name in ("gold.html", "concept.html", "membership.html", "manifest.json"):
            shutil.copy2(os.path.join(ANNOTATE, name), os.path.join(args.publish, name))
        copied = publish_images(args.publish, [item["id"] for item in gold + concept + membership])
        index = os.path.join(args.publish, "index.html")
        if True:
            total = len(gold) + len(concept) + len(membership)
            with open(index, "w", encoding="utf-8") as handle:
                handle.write(f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CS5495 课程项目 · 图像标注</title>
<style>
  body {{ margin: 0; background: #14171c; color: #e8eaed; line-height: 1.65;
         font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }}
  main {{ max-width: 820px; margin: 0 auto; padding: 40px 20px 80px; }}
  h1 {{ font-size: 24px; margin: 0 0 6px; }}
  h2 {{ font-size: 17px; margin: 34px 0 10px; color: #cfd6df; }}
  p, li {{ font-size: 14px; color: #c2cad4; }}
  a {{ color: #7fb3ff; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13.5px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #262d36; vertical-align: top; }}
  th {{ color: #9aa4b2; font-weight: 600; }}
  code {{ background: #222933; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
  .tag {{ display: inline-block; font-size: 12px; padding: 1px 7px; border-radius: 999px;
          background: #22303f; color: #8fb8ff; margin-left: 6px; }}
  .note {{ background: #1d2229; border: 1px solid #2b323c; border-radius: 8px;
           padding: 12px 14px; margin: 14px 0; }}
</style>
</head>
<body>
<main>
  <h1>多模态仇恨言论检测模型的图像标注</h1>
  <p>CS5495 Explainable AI 课程项目 · 内部标注页 <span class="tag">不公开索引</span></p>

  <h2>这个项目在做什么</h2>
  <p>被审计的模型是 <code>Amirhossein75/clip-vit-base-mmhs150k-mtl</code>：CLIP 的图像塔与文本塔，
  加一个门控融合和五个类别头，输入是「推文文字 + 配图」。模型的准确率已经验证过与作者公布的一致，
  所以问题不是「它准不准」，而是<b>它到底有没有用图</b>。</p>
  <p>现有结论是文字主导：模态 Shapley 分解里文字平均贡献 +0.047，图像只有 +0.0014，
  文本主导 99.4% 的样本。这个结论要成立，必须先排除三种解释方式上的伪影，
  而这三件事都只能靠人眼判断，机器判不了。</p>

  <h2>需要标注什么</h2>
  <table>
    <thead><tr><th>页面</th><th>数量</th><th>判断什么</th><th>用来做什么</th></tr></thead>
    <tbody>
      <tr><td><a href="gold.html">gold.html</a></td><td>{len(gold)}</td>
          <td>图里有没有可读的文字</td>
          <td>OCR 金标准：算出 EasyOCR 的检出率与误检率，说明分组误差的方向</td></tr>
      <tr><td><a href="concept.html">concept.html</a></td><td>{len(concept)}</td>
          <td>图里有没有可读的文字</td>
          <td>概念集 QC：末尾 100 张是盲复标，用来算标注一致率（门槛 0.8）</td></tr>
      <tr><td><a href="membership.html">membership.html</a></td><td>{len(membership)}</td>
          <td>这张图是否属于页面上写的那个概念</td>
          <td>概念归属抽检：概念集是检测器自动生成的，这一步测它的敏感度与特异度</td></tr>
    </tbody>
  </table>
  <p>三个页面合计 <b>{total}</b> 次判断。全部做完约 1.5–2 小时；只做 gold 与 membership（{len(gold) + len(membership)} 次）
  约 30 分钟，也够支撑报告里需要人工数据的两处结论。</p>

  <h2>怎么标</h2>
  <ol>
    <li>先点开一个页面，在右上角填<b>标注者代号</b>，导出的文件名会带这个名字。</li>
    <li>键盘 <code>1</code> / <code>2</code> / <code>3</code> 选择，<code>←</code> 回退到上一张，选完自动跳下一张。</li>
    <li>进度存在浏览器本地，关掉页面再打开会从第一张没标的继续。</li>
    <li>标完点<b>「导出结果」</b>，把下载的 JSON 发回给项目组，放进 <code>annotate/out/</code> 后由脚本合并。</li>
  </ol>
  <div class="note">
    <b>两点要求。</b>一是按第一眼判断，不要放大反复找；二是不要顺着某一类图的规律去猜，
    页面故意不显示检测结果和样本来源，就是为了让判断独立。
  </div>

  <h2>关于数据</h2>
  <p>图片来自公开数据集 MMHS150K，内容是真实的推文配图，包含仇恨言论与冒犯性画面。
  标注结果只用于这次课程作业的统计分析，不用于训练模型，不对外发布，不进入任何公开数据集。</p>
  <p>这个站点是公开可访问的 GitHub Pages，没有登录。链接不要转发到公开渠道；
  页面加了 <code>noindex</code>，不会被搜索引擎收录，但拿到链接的人都能打开。</p>
</main>
</body>
</html>
""")
        img_dir = os.path.join(args.publish, "img")
        total = sum(os.path.getsize(os.path.join(img_dir, name)) for name in os.listdir(img_dir))
        print(f"publish: {copied} new images, {total / 1e6:.0f} MB total in {args.publish}/img")


if __name__ == "__main__":
    main()
