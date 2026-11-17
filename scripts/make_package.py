"""Build the hand-off package: notebook, code, results, docs, annotation exports.

Excluded on purpose: data/ (3.3 GB of images + the 49 MB GT dump), publish/ (the
GitHub Pages copy of the images), logs/, .git, __pycache__. Everything a reader needs
to follow the report or cut the video is inside; everything needed to re-run is named in
the README.

Usage: python scripts/make_package.py
"""

from __future__ import annotations

import os
import shutil
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE = os.path.join(os.environ.get("TEMP", "."), "cs5495_package")
ZIP = os.path.join(ROOT, "CS5495-CourseProject-Package-20261114.zip")

INCLUDE_DIRS = ["scripts", "src", "figures", "refine-logs", "annotate", "templates"]
INCLUDE_FILES = [
    "Course Project-2026A.ipynb",
    "Course Project-2026A.pdf",
    "Course Project-2026A.html",
    "refine-logs/EXPERIMENT_PLAN.md",
    "refine-logs/EXPERIMENT_TRACKER.md",
    "findings.md",
    "Project Proposal (EN).md",
    "Project Proposal (EN).pdf",
    "Project Proposal (CN).md",
]

README = """# CS5495 课程项目 · 交接包

多模态仇恨言论检测模型的 XAI 忠实性审计。被审计模型 `Amirhossein75/clip-vit-base-mmhs150k-mtl`
（CLIP ViT-B/32 双塔 + 门控融合 + 五头）。核心问题：**模型到底有没有用图，说明方法能不能信。**

## 先看什么

| 文件 | 用途 |
|---|---|
| `Course Project-2026A.pdf` | 最终报告，25 页，所有图表已排版 |
| `Course Project-2026A.ipynb` | 同一份报告的可执行版本，37 个单元，已带输出 |
| `Course Project-2026A.html` | 不用 Jupyter 也能直接读 |
| `TEAM_SUMMARY.md` | **做总结和视频先读这个**：结论、关键数字、章节地图、60 秒分镜 |
| `refine-logs/EXPERIMENT_TRACKER.md` | P0–P8 每一行的状态、结果摘要、偏差与事故记录 |
| `findings.md` | 按时间追加的发现日志，包括被推翻的早期结论 |

## 目录

```
scripts/        所有实验脚本（p0_* … p8_*）+ 报告构建 + 标注工具
src/mmhs/       模型加载、归因方法、干预与忠实性度量
refine-logs/    结果 JSON / JSONL + 计划与追踪表
figures/        报告用图 fig1–fig4
annotate/       三个标注页面、manifest、人工标注导出（out/）
templates/      课程发的原始 notebook 模板（未改动）
```

## 结果文件怎么读

每个实验写一个 JSON，字段与报告里的数字一一对应：

| 文件 | 内容 |
|---|---|
| `p0_gate.json` | 复现闸门：五头 AUC/F1 与发布值对照 |
| `p0_ocr.jsonl` / `p0_ocr_stats.json` | 3,499 张图的 EasyOCR 框 |
| `p0_text_gold.json` | 200 张盲标金标准：检出率、误检率、κ |
| `p1_matrix.json` / `p1_decision.json` | 七方法 AOPC、赢家与平局判定 |
| `p1_sanity_curves.json` | 随机化闸门（逐层 ρ、全随机 r） |
| `p1_stability.json` / `p1_agreement.json` | 种子/噪声稳定性、方法间一致性 |
| `p1d_*_attr_test.jsonl` | 赢家在 3,000 条上的逐样本归因 |
| `p2_phi.jsonl` / `p2_summary.json` / `p2_curves.json` | 模态 Shapley、删除曲线 |
| `p3_overlap.json` | 图内文字重叠检验（负结果） |
| `p3_tcav.json` / `p3_concept_audit.json` / `p3_concept_text.json` | TCAV、概念标签质量 |
| `p4_flip.json` / `p4_qualitative.json` | 消融翻转率、40 条定性编码 |
| `p5_agreement.json` | 人机一致性（未过门槛） |
| `p6_roar.json` | 模态级 ROAR 三臂 |
| `p7_siglip2_phi.json` | SigLIP2 checkpoint 稳健性 |
| `p8_robustness.json` | 预处理消融、四头/五头敏感性 |

## 重新运行

同样的内容也在 GitHub：`https://github.com/Botherain/cs5495-mmhs-xai`（多带
`TEAM_SUMMARY.md`、`CONTRIBUTION.md` 与 README 里的结论链一节）。二选一即可。

数据不在包内（3.3 GB 图片 + 49 MB GT）。复现需要：

1. 把 `MMHS150K_GT.json`、`subset_ids.json`、`img_resized/` 放回 `data/MMHS150K/`；
2. 模型权重放到 `C:\\Users\\123\\AppData\\Local\\Temp\\opencode\\mmhs_model`（或设 `MMHS_MODEL_DIR`）；
3. 按 `refine-logs/EXPERIMENT_TRACKER.md` 的行序跑脚本，每个脚本都能断点续跑；
4. `python scripts/build_notebook.py` 重建 notebook，`python scripts/export_pdf.py` 导 PDF。

notebook 本身只读结果文件，不加载模型（§2 的干预演示除外），所以不装 CUDA 也能完整执行。

## 环境

Python 3.14、PyTorch 2.11 (CUDA 12.8)、transformers 5.17、easyocr 1.7.2、RTX 5080。
"""

TEAM_SUMMARY = """# 给队友的总结与视频素材

## 一句话

这个模型几乎不读图：文字贡献 +0.047，图像 +0.0014，99.4% 的样本由文字主导；而且当我们
用三种不同方法追问"图像那一点点贡献到底是什么"时，三种方法都给出否定或不可用的答案。

## 二、关键数字（报告里都能查到，可直接背）

| 结论 | 数字 | 位置 |
|---|---|---|
| 复现闸门通过 | 五头 macro ROC 0.7759（发布 0.7832）、F1 0.5660（发布 0.5690） | §1 |
| 七方法中只有两个在文本上可信 | LIME +0.0620、KernelSHAP +0.0606，均约为门槛 6 倍 | §3 |
| 图像侧只有 occlusion 过门槛 | +0.0013，Grad-CAM −0.0010、IG −0.0003、Score-CAM −0.0001 | §3 |
| 图像梯度方法描述的是架构不是函数 | 视觉塔全随机化后 Grad-CAM 仍与训练模型 ρ=0.68 | §4 |
| 文本主导 | φ_T +0.0467 vs φ_I +0.0014；主导率 99.4% | §5 |
| 图像经常"帮倒忙" | 56.5% 的样本 φ_I < 0 | §5 |
| 归因确实在动决策 | 删 top-10% 文本单元翻转 38.8%，随机 11.1%（Δ27.7 pp） | §7 |
| 但图内文字没有超额归因 | 7.27% vs 等面积对照 7.75%，−0.48 pp，p=2.1e−07 | §6 |
| 概念探针无法判定 | 梯度共线度 0.830，TCAV 分数饱和为 0/1 | §6 |
| 强制用图反而更差 | dropout 臂图像份额 −11.4 pp，F1 +0.022（反而更准） | §8 |
| 人机一致性未过门槛 | ρ=−0.071（要求 −0.20），控制投票数与目标水平后 −0.045 | §9 |
| OCR 覆盖率是下界 | 检出 70.9%、误检 0%，真实含字率约 51.6%（人标 51.8%） | §1 |

## 三、章节地图（写总结时按这个顺序讲）

- **§0–§2** 问题、模型、单位与"忠实"的定义。关键设计：删除只改 attention mask，不改 ids。
- **§3** 七个候选方法先比忠实性，再谈模型结论，这是课程要求的顺序。
- **§4** 随机化闸门：把塔的层随机化，看解释是否跟着塌。图像梯度方法在这里第二次失败。
- **§5** 两模态 Shapley，报告的主结果。
- **§6** 追查图像贡献的内容：重叠检验 + TCAV + 标签审计，全部负结果。
- **§7** 因果检验：删掉被指的单元，标签会不会变；配 40 条人工检视。
- **§8** 训练层面干预（ROAR），试图让模型用图，失败。
- **§9** 人机一致性，未过门槛。
- **§10** 三项稳健性：预处理、四头/五头、SigLIP2。
- **§11** 九个负结果与事故，包括一个会让 §3 结论反转的 bug。
- **§12–§13** 结论与复现清单。

## 四、四个必须讲清的负结果

1. **图内文字重叠为负**：归因落在文字上的份额略低于对照，方向相反。原先"图像在重读图里的字"的假设被自己的实验否掉。
2. **TCAV 退化**：逐样本梯度共线度 0.83，符号统计只能出 0 或 1；标签最好的概念探针也只有 0.62 准确率。
3. **ROAR 反向**：加 dropout 后图像份额下降，不是上升；机制是 presence 仍为 1，模型学到"这个向量没用"。
4. **人机一致性未过**：方向对、显著，但 |ρ|=0.10 < 0.20，按未通过报告。

## 五、60 秒视频分镜

| 时间 | 画面 | 旁白（要点） |
|---|---|---|
| 0–6 s | 一张推文配图快速推近 | "一个能看图的仇恨言论检测器，它真的在看图吗？" |
| 6–16 s | 模型结构示意 + 闸门表格 | 模型复现了发布指标，所以我们审的是原模型，不是残缺副本。 |
| 16–28 s | **fig2_phi.png**（左右两幅直方图） | 文字 Shapley +0.047，图像 +0.0014。99.4% 的样本，文字说了算。 |
| 28–38 s | **fig3_deletion.png** + 翻转率数字 | 而且图像常常帮倒忙：56.5% 的样本它的贡献是负的；删掉图像上最该负责的格子，3000 条里只翻转了 5 条。 |
| 38–50 s | **fig1_aopc.png** + TCAV 表 | 反过来问"那图像在看什么"：三种方法都答不上来：归因避开图里的文字，概念探针的梯度几乎不变。 |
| 50–60 s | **fig4_gates.png** 收尾 | 连强制训练（dropout）都让它更依赖文字。结论：报告 per-modality 消融，别只看 macro-F1 0.65。 |

素材：`figures/fig1_aopc.png`、`fig2_phi.png`、`fig3_deletion.png`、`fig4_gates.png`。
需要单页大图就按 § 号在 PDF 里截。

## 六、别踩的坑

- "图像被忽略"不准确：它是**负贡献**居多，不等于不参与。
- 误差分类学里的 `false_positive` 是按严格多数集合定义的，1,956/3,000，不要写成"二分类误报"。
- `religion` 头被排除在目标外（AUC 0.605、2.7% 患病率），五头数字只出现在闸门表。
- ROAR 的重训只用了 2,000 条，绝对 F1 不可与原模型比，只能比 arm2 与 arm3。
- SigLIP2 的 φ 绝对值不可用（空白基线在该 checkpoint 上偏高），只讲主导率。
"""

CONTRIBUTION = """# 贡献表（50 / 50）

课程要求提交 50/50 的分工说明。下面是本项目的工作项清单，**归属一栏请两位自行确认后再提交**，
不要直接沿用这里的占位。

| 工作项 | 内容 | 归属 |
|---|---|---|
| 提案 | 研究问题、方法选型、预登记判据 | 待确认 |
| 数据管线 | 3,499 张图抓取、抽样、OCR、金标准标注页 | 待确认 |
| 模型与闸门 | checkpoint 复现、五头指标、阈值口径 | 待确认 |
| 归因方法 | 3 文本 + 4 图像方法、融合路径复刻 | 待确认 |
| 忠实性评估 | AOPC、随机化闸门、稳定性、方法一致性 | 待确认 |
| 模态 Shapley | 四联盟、交互项、删除曲线 | 待确认 |
| 图像证据追查 | 重叠检验、TCAV、概念标签审计 | 待确认 |
| 失败模式 | 消融翻转、误差分类、40 条定性检视 | 待确认 |
| 人机一致性 | 一致性分桶、相关检验 | 待确认 |
| ROAR 与 SigLIP2 | 三臂重训、跨 checkpoint 稳健性 | 待确认 |
| 报告与图表 | notebook、PDF、四张图 | 待确认 |
| 标注执行 | 2,724 次判断 + 40 条定性编码 | 待确认 |
| 视频 | 脚本、录制、剪辑 | 待确认 |

最终提交按 **50% / 50%** 计。
"""


def main() -> None:
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    copied = 0
    for name in INCLUDE_DIRS:
        source = os.path.join(ROOT, name)
        if not os.path.isdir(source):
            continue
        shutil.copytree(source, os.path.join(STAGE, name),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))
        copied += 1
    for name in INCLUDE_FILES:
        source = os.path.join(ROOT, name)
        if os.path.exists(source):
            target = os.path.join(STAGE, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)
            copied += 1

    for name, text in (("README.md", README), ("TEAM_SUMMARY.md", TEAM_SUMMARY),
                       ("CONTRIBUTION.md", CONTRIBUTION)):
        with open(os.path.join(STAGE, name), "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for base, _, files in os.walk(STAGE):
            for name in files:
                path = os.path.join(base, name)
                archive.write(path, os.path.relpath(path, STAGE))

    size = os.path.getsize(ZIP) / 1e6
    with zipfile.ZipFile(ZIP) as archive:
        entries = archive.namelist()
    print(f"staged {copied} items, zip entries {len(entries)}, {size:.1f} MB -> {ZIP}")


if __name__ == "__main__":
    main()
