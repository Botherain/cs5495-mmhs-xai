# CS5495 · 多模态仇恨言论检测模型的 XAI 忠实性审计

这个仓库是一个课程项目的完整复现包：审计 `Amirhossein75/clip-vit-base-mmhs150k-mtl`
（CLIP ViT-B/32 双塔 + 门控融合 + 五个分类头）到底有没有使用它的图像模态，以及在回答这个
问题之前，七种解释方法里哪些本身可信。

**一句话结论。** 模型复现了自己的发布指标，但决策 99.4% 由文字承载；追问图像那 0.1% 的贡献
"是什么"时，三种独立探针全部失败，强行训练它用图反而让准确率下降、图像份额更低。

- 报告：`Course Project-2026A.pdf`（25 页）、`Course Project-2026A.ipynb`（37 单元，已执行）
- 结果：`refine-logs/`（每个实验一个 JSON，字段与报告数字一一对应）
- 计划与追踪：`refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_TRACKER.md`
- 发现日志（含被推翻的早期结论）：`findings.md`

---

## 1. 研究问题

数据是 MMHS150K：推文文字 + 配图 + 三名标注者的仇恨类别标签。被审模型把两个 CLIP 塔的输出
投影后做门控融合，再经五个 sigmoid 头（racist / sexist / homophobe / religion / otherhate）
输出。

课程的要求决定了提问顺序：**先证明解释方法本身忠实，再用它谈模型**。所以项目实际问了三个问题：

1. 七种候选归因方法（文本 3 种、图像 4 种），哪些在这个 checkpoint 上是忠实的？
2. 在忠实的那些方法下，两个模态各承担多少决策？
3. 图像承担的那一点点，内容是什么？

## 2. 设计原则

这四条决定了后面所有实验的形状，读代码前值得先看：

1. **先验判据**：AOPC 门槛（文本 0.01、图像 0.001）、随机化闸门（ρ_depth < −0.7 且 r_full < 0.2）、
   稳定性（ρ > 0.5）、翻转率差（> 5 pp）等全部写在实验计划里，跑之前就固定，事后不改。
2. **单变量干预**：文本删除只把 `attention_mask` 置 0，不动 `input_ids`（CLIP 在第一个 EOS 处
   池化，改 id 会让池化位置前移，等于同时改了两件事）；图像删除写均值色，与联盟博弈里的空白
   图像一致。
3. **空白模态是真实输入**：文本用空字符串的编码，图像用均值色图，不是零向量（零向量对
   LayerNorm 是域外的）。
4. **每个结论至少两个独立仪器**：AOPC 对随机化闸门，Shapley 对删除曲线，翻转率对人工编码。
   任何一条单独成立都不写进结论。

## 3. 快速开始

### 只想读报告

打开 `Course Project-2026A.pdf`。不需要装任何东西。

### 想跑 notebook（不加载模型也能跑）

notebook 只读结果文件，除 §2 的一个干预演示外不碰模型：

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 "Course Project-2026A.ipynb"
```

### 想从头复现全部数字

```bash
# 1. 数据：从 6 GB 的官方 zip 里按成员下载，只取需要的 3,499 张
python scripts/fetch_subset.py --ids data/MMHS150K/subset_ids.json

# 2. 模型（自定义 modeling 代码，需要 trust_remote_code）
huggingface-cli download Amirhossein75/clip-vit-base-mmhs150k-mtl --local-dir models/mmhs
set MMHS_MODEL_DIR=%CD%\models\mmhs        # PowerShell: $env:MMHS_MODEL_DIR="..."

# 3. 按 EXPERIMENT_TRACKER.md 的行序跑；每个脚本都能断点续跑
python scripts/p0_gate.py
python scripts/p0_ocr.py
python scripts/p1_text_attributions.py
...
python scripts/build_notebook.py
python scripts/export_pdf.py
```

预计耗时（RTX 5080）：数据下载 30–60 min，OCR 25 min，P1 归因 40 min，P1.7 两模态 30 min，
P2 联盟 20 min，P4 翻转 30 min，其余各 5–15 min。总计约 4 小时。

## 4. 数据与模型

| 资产 | 来源 | 说明 |
|---|---|---|
| MMHS150K | `http://datasets.cvc.uab.es/MMHS150K/MMHS150K.zip` | 6 GB；`scripts/fetch_subset.py` 用 HTTP range 只取需要的成员 |
| `subset_ids.json` | 本仓库 `data/MMHS150K/` | **报告里所有数字都是在这 3,000 + 499 条上算的**，务必用它，不要重新抽样 |
| CLIP checkpoint | HF `Amirhossein75/clip-vit-base-mmhs150k-mtl` | 主审计对象 |
| SigLIP2 checkpoint | HF `Amirhossein75/siglip2-base-mmhs150k-fusion` | §10 的跨 checkpoint 检查 |

`fetch_subset.py` 不带 `--ids` 时会重新分层抽样，得到统计上可比但**不是同一批**的样本；
要复现报告请始终带 `--ids`。

数据目录结构：

```
data/MMHS150K/
  MMHS150K_GT.json        标签与原文（49 MB，下载脚本自动取）
  subset_ids.json         样本清单（随仓库提交）
  img_resized/<id>.jpg    3,499 张图
  splits/{train,val,test}_ids.txt
```

## 5. 目录

```
scripts/            实验脚本，文件名即阶段号（p0_* … p8_*）
src/mmhs/           模型加载、归因方法、干预与忠实性度量
refine-logs/        结果 JSON/JSONL + 实验计划 + 追踪表
figures/            报告四张图
annotate/           人工标注页面与导出（3 个队列 + 40 条定性检视）
templates/          课程原始 notebook（未改动）
data/MMHS150K/      仅提交 subset_ids.json，图片与 GT 需自行下载
```

`src/mmhs/` 三个文件：

- `model.py`：加载 checkpoint（`trust_remote_code`）、`Wrapper`（编码、单头/多头概率、
  门控统计）、`USABLE_HEADS = [0,1,2,4]`（剔除 religion）。
- `attribution.py`：融合路径复刻（不经过塔，直接吃 pooler 特征）、文本 3 方法与图像 4 方法、
  `content_positions`（排除 CLIP 的首尾哨兵）。
- `faithfulness.py`：AOPC 曲线、删除/插入干预。

## 6. 实验设计与代码对照

| 阶段 | 问题 | 干预 / 度量 | 代码 | 产出 | 预登记判据 |
|---|---|---|---|---|---|
| P0 | 这个 checkpoint 是它声称的那个模型吗？ | 重算五头 AUC/F1 对发布值 | `p0_gate.py` | `p0_gate.json` | AUC 差 <0.03、F1 高于下限 0.539 |
| P0 | 图里有多少可读文字？ | EasyOCR ≥0.6 + 200 张盲标金标准 | `p0_ocr.py`、`annotate/gold.html` | `p0_ocr.jsonl`、`p0_text_gold.json` | 检出率/误检率带 Wilson 区间 |
| P1 | 七种归因方法哪些忠实？ | AOPC（删/插平均）、comprehensiveness、sufficiency，对随机排序配对检验 | `p1_text_attributions.py`、`p1_image_attributions.py`、`p1_evaluate.py`、`p1_decision.py` | `p1_matrix.json`、`p1_decision.json` | 文本 0.01 / 图像 0.001，且优于随机 |
| P1 | 忠实性会不会只是描述了架构？ | 逐层随机化塔（0/3/6/9/12 块），看归因是否跟着塌 | `p1_sanity_stability.py --stage gates` | `p1_sanity_curves.json` | ρ_depth < −0.7 且 r_full < 0.2 |
| P1 | 归因稳定吗？ | 3 个种子 / 5% 像素噪声，取最差一对 | `p1_sanity_stability.py --stage stability` | `p1_stability.json` | ρ > 0.5 |
| P1 | 方法之间一致吗？ | 全选择集上逐样本 Spearman 与 top-10% 重叠 | `p1_sanity_stability.py --stage agreement` | `p1_agreement.json` | 描述性 |
| P1 | 赢家在全体样本上长什么样？ | 赢家方法在 3,000 条上重算 | `p1d_winner_3k.py` | `p1d_*_attr_test.jsonl` | — |
| P2 | 模态各占多少？ | 四联盟 Shapley（空白 = 空文本 + 均值色图） | `p2_modality.py` | `p2_phi.jsonl`、`p2_summary.json`、`p2_curves.json` | 完备性误差 <1e−6；主导率与 CI |
| P3 | 图像归因落在图内文字上吗？ | 面积加权重叠 vs 等面积随机框（100 次） | `p3_overlap.py` | `p3_overlap.json` | 超额 > 15 pp |
| P3 | 图像塔编码了什么概念？ | TCAV（768-d pooler 梯度）+ 20 个随机方向 | `p3_tcav.py`、`p3_concepts.py` | `p3_tcav.json`、`p3_concepts.json` | 超随机方向且 split-half 一致 |
| P3 | 概念标签本身可信吗？ | 每概念 100 正 + 100 负盲标 | `annotate/membership.html`、`merge_annotation.py` | `p3_concept_audit.json` | 敏感度/特异度 > 0.8 |
| P4 | 被指的单元真的承载决策吗？ | 删 top-10% vs 随机删同等数量，McNemar | `p4_flip.py` | `p4_flip.json` | 翻转率差 > 5 pp |
| P4 | 删掉的到底是什么？ | 20 翻转 + 20 对照逐条编码 | `annotate/qualitative.html`、`merge_qualitative.py` | `p4_qualitative.json` | 描述性 |
| P5 | 模型和人看同样的线索吗？ | 标注一致性 vs 文本份额，单侧 Spearman | `p5_agreement.py` | `p5_agreement.json` | ρ < −0.2 |
| P6 | 强制它用图会更好吗？ | 融合头三臂重训（原版 / 重训 / 重训+模态 dropout） | `p6_roar.py` | `p6_roar.json` | 图像份额上升且 F1 降幅 ≤0.01 |
| P7 | 换个 checkpoint 还成立吗？ | SigLIP2 + 同样的融合头，重算 Shapley | `p7_siglip2.py` | `p7_siglip2_phi.json` | 主导率差 <10 pp |
| P8 | 结论依赖预处理 / 目标定义吗？ | 去 URL 重算；四头 vs 五头 | `p8_robustness.py` | `p8_robustness.json` | 位移 <10%；主导率差 <5 pp |

## 7. 结论链：从哪个数字推出哪句话

报告按下面这条链组织，每一步都依赖上一步成立：

**第 1 步 · 模型是它自己。** 五头 macro ROC 0.7759（发布 0.7832）、macro F1 0.5660
（发布 0.5690、下限 0.539），判定 REPRODUCED。→ 之后所有的异常都不能用"权重坏了"解释。

**第 2 步 · 先筛方法，再谈模型。** 七种方法里只有 LIME（+0.0620）、KernelSHAP（+0.0606）
在文本上越过 0.01 门槛，图像只有 occlusion（+0.0013）越过 0.001。梯度类图像方法（Grad-CAM
−0.0010、IG −0.0003、Score-CAM −0.0001）低于随机。→ 后面只用这三个赢家解释模型。

**第 3 步 · 随机化闸门给出第二个独立理由。** 把视觉塔逐层随机化到全随机后，Grad-CAM 的归因
与训练模型仍有 ρ=0.68 的相关。一个解释如果连随机网络都"解释"得一样，它描述的是架构，与学到的函数无关。→ 图像梯度方法被两条独立证据否掉两次。

**第 4 步 · 主结果：文字主导，且图像经常反向。** 3,000 条上 φ_T = +0.0467、φ_I = +0.0014，
文字主导 99.4% 的样本，完备性误差 6.9e−08；φ_I 在 56.5% 的样本上为负。→ 正确的表述是
"图像被用得很少，而且常常与文字证据相反"，不是"图像被忽略"。

**第 5 步 · 追问图像那一点贡献是什么，三种探针全部失败。**
   - 归因落在图内文字上吗？观测 7.27% vs 等面积对照 7.75%，**−0.48 pp**（p=2.1e−07）：
     落在文字上的份额略低于对照，方向相反。
   - 概念探针能给出方向吗？不能：逐样本梯度共线度 **0.830**，符号统计只能输出 0/1；
     而且概念集是检测器生成的，face 负例 44% 实际含脸。
   - 强制训练能让它用图吗？不能：dropout 臂图像份额 **−11.4 pp**、F1 +0.022（反而更准）。
   → 三件事互不依赖，结论只能是"图像贡献的内容至今没有证据"。

**第 6 步 · 因果检验确认归因不是描述输入。** 删掉 top-10% 文本单元翻转 38.8% 的决策，
随机删除只有 11.1%（Δ27.7 pp，p=5.3e−254）；人工检视 20 条翻转，20/20 删掉的都含侮辱词，
对照臂只有 9/20、且有 10/20 落在 URL 碎片上。→ 文本归因确实指向承载标签的词。
同一操作在图像上只翻转 5/3,000 条，与 φ_I ≈ 0.0014 一致。

**第 7 步 · 人机一致性：方向对、力度不够。** ρ = −0.071（要求 −0.20）；控制投票数与目标水平后 −0.045，显著但未过门槛。
分桶显示一致性越低、φ_T 越大，模型在歧义样本上更多看文字，图像份额没有上升。
→ 报告按未通过处理，并解释为"歧义是词汇层面的，不是视觉层面的"。

**第 8 步 · 排除三个替代解释。** 去 URL/@/RT 后 φ_T 只移动 3.1%（ρ=0.961）；四头与五头
主导率 99.4% vs 96.4%；SigLIP2 上主导率 94.8%。→ 结论不是预处理、目标定义或某一个
checkpoint 的产物。

## 8. 结果文件 ↔ 报告章节

| 报告 | 结果文件 |
|---|---|
| §1 模型与数据 | `p0_gate.json`、`p0_stats.json`、`p0_ocr_stats.json`、`p0_text_gold.json` |
| §3 方法筛选 | `p1_matrix.json`、`p1_decision.json`、`p1_faithfulness.jsonl` |
| §4 随机化/稳定性 | `p1_sanity_curves.json`、`p1_stability.json`、`p1_agreement.json` |
| §5 模态 Shapley | `p2_summary.json`、`p2_phi.jsonl`、`p2_curves.json` |
| §6 图像内容 | `p3_overlap.json`、`p3_tcav.json`、`p3_concepts.json`、`p3_concept_audit.json`、`p3_concept_text.json` |
| §7 失败模式 | `p4_flip.json`、`p4_qualitative.json` |
| §8 ROAR | `p6_roar.json` |
| §9 人机一致性 | `p5_agreement.json` |
| §10 稳健性 | `p8_robustness.json`、`p7_siglip2_phi.json` |
| §11 负结果 | 上述文件的失败项 + `findings.md` |

## 9. 复现顺序

严格按 `EXPERIMENT_TRACKER.md` 的行序；依赖关系是：

```
p0_gate ─┬─> p1_*  ──> p1d_*  ─┬─> p3_overlap
p0_ocr  ─┘                      ├─> p4_flip ──> p4_qualitative
                                └─> p2_modality ──> p5_agreement
p3_concepts ──> p3_tcav
p0_ocr + p2_phi ──> p6_roar
p2_phi ──> p7_siglip2
p2_phi ──> p8_robustness
所有 JSON ──> build_notebook.py ──> export_pdf.py
```

每个长脚本都逐样本写 JSONL 并在重启时跳过已完成 id，中断最多损失一条样本。

## 10. 已知限制

写总结或答辩时这六条必须知道：

1. **样本**：3,000 条分析集与 499 条选择集都取自官方 test split，但抽样不是官方划分本身。
2. **`religion` 头被排除**：AUC 0.605、患病率 2.7%，太弱无法解释；五头数字只出现在闸门表。
3. **误差分类学**：`false_positive` 按严格多数集合定义，1,956/3,000 不代表二分类误报率。
4. **ROAR 重训只有 2,000 条**：绝对 F1（0.38）不可与原模型（0.58）比，只能比 arm2 与 arm3。
5. **SigLIP2 的 φ 绝对值不可用**：空白基线在该 checkpoint 上高于真实输入（φ 双负），只讲主导率。
6. **OCR 召回 70.9%**：按 EasyOCR 检出分组是真实含字集的子集，反向无污染（误检 0%）。
7. **概念集是检测器生成的**：计划要求人工筛选，实际用三个检测器，偏差记录在
   `EXPERIMENT_TRACKER.md`；事后用 600 张盲标量化了标签质量。

## 11. 人工标注

三个队列共 2,724 次判断，页面在 `annotate/`（单人本地打开即可，图片走相对路径）：

| 页面 | 数量 | 判断 | 产出 |
|---|---|---|---|
| `gold.html` | 200 | 图里有没有可读文字（OCR 金标准） | `p0_text_gold.json` |
| `concept.html` | 1,924 | 同上（1,824 + 100 盲复标） | `p3_concept_text.json` |
| `membership.html` | 600 | 是否属于该概念（每概念 100 正 + 100 负） | `p3_concept_audit.json` |
| `qualitative.html` | 40 | 被删片段是什么、翻转是否合理 | `p4_qualitative.json` |

导出文件放在 `annotate/out/`，`python scripts/merge_annotation.py` 与
`python scripts/merge_qualitative.py` 合并；多人标同一队列时按多数取值，平票丢弃并计数。

## 12. 环境

Python 3.14.5、PyTorch 2.11.0+cu128、transformers 5.17.0、numpy 2.5.3、scipy 1.18.1、
pandas 3.0.5、scikit-learn 1.9.0、matplotlib 3.11.1、opencv 4.11.0、easyocr 1.7.2、
nbformat 5.11.1、nbconvert 7.17.1、RTX 5080。

transformers 5.x 下 SigLIP2 的 `get_text_features` / `get_image_features` 返回
`BaseModelOutputWithPooling`，`scripts/p7_siglip2.py` 取 `.pooler_output`；模型自带的
forward 直接把该对象喂进 `F.normalize`，所以原实现无法在新版本上运行，仓库里用 pooler 张量
重实现了融合块，并与自身 forward 做过逐样本对拍（最大差 0.0）。

## 13. 数据与许可

MMHS150K 为研究用途公开数据集，本仓库不重分发任何图片或推文原文，只提交样本 id、统计量与
结果 JSON。标注页面会把原始推文与图片并排显示，属于仇恨内容，未部署到公网。
