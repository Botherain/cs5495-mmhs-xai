# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | 环境与权重 sanity | clip-vit-base MTL（本地方案） | 100 条 | 前向可用、参数量 153M | MUST | IN-PROGRESS | 需 shim：v5 不兼容，key 重映射 |
| R002 | M0 | 数据落盘与完整性 | MMHS150K 官方包 | 全量 | 图像覆盖率、GT 条数 | MUST | TODO | 6 GB 下载中 |
| R003 | M0 | 基线复现 | 同上 | test 子集 | macro-F1、macro ROC-AUC | MUST | TODO | 目标 0.569 / 0.783 |
| R004 | M0 | IAA 统计与分层抽样 | — | 全量 → 3,000 | hate 占比、三档一致性比例 | MUST | TODO | 固定种子 |
| R005 | M1 | 文本侧 AOPC | LIME / KernelSHAP / Occlusion | 500 | AOPC-del、AOPC-ins | MUST | TODO | 删除=替换 |
| R006 | M1 | 图像侧 AOPC | Grad-CAM / IG / Occlusion / Score-CAM | 500 | AOPC-del、AOPC-ins | MUST | TODO | SLIC 超像素 |
| R007 | M1 | Sanity check | 8 方法 × 12 层随机化 | 500 | Spearman（原 vs 随机化） | MUST | TODO | 门槛 |
| R008 | M1 | 稳定性与一致性 | 8 方法 | 500 | 种子/噪声 Spearman、top-k 重叠 | MUST | TODO | 佐证 |
| R009 | M2 | 模态 Shapley | 4 联盟 × 2 基线 | 3,000 | φ_T、φ_I、完备性误差 | MUST | TODO | 空白输入 / flag 置零 |
| R010 | M2 | 交互指数 | 同上 | 3,000 | I 值分布、二项检验 | MUST | TODO | 复用同批前向 |
| R011 | M2 | 模态删除曲线 | 真实排序 vs 随机 | 3,000 | 模态级 AOPC、配对 t | MUST | TODO | — |
| R012 | M2 | 门控对照 + 跨层级一致性 | 门控范数比、token 归因总和 | 3,000 | Spearman | MUST | TODO | ρ≈0 也算发现 |
| R013 | M3 | OCR vs 背景遮挡 | 40 条案例 | 40 | φ_I 变化量 | MUST | TODO | 人工挑图 |
| R014 | M3 | PFI / PDP | 分组置换、bootstrap | 500 | null 分位、置信带 | MUST | TODO | B=100 / 200 |
| R015 | M3 | TCAV | 3 概念 × 20 随机方向 | 500 | p 值、split-half 相关 | MUST | TODO | 概念集 11–13 Nov |
| R016 | M3 | 人类一致性分析 | 三档分层 | 3,000 | 置信度均值/方差、Kruskal-Wallis | MUST | TODO | RQ5 |
| R017 | M4 | 特征缓存 | 冻结编码器 | 3,000 | — | MUST | TODO | 供 R018 复用 |
| R018 | M4 | head-only 重训 + 模态 dropout | 基线头 vs dropout 头 | 3,000 | 子集召回、macro-F1 | MUST | TODO | 3 seeds |
| R019 | M4 | 翻转率检验 | top-k vs 随机消融 | 500 | 翻转率差、配对检验 | MUST | TODO | Stage 5 判据 |
| R020 | M2/M3 | SigLIP2 复现 | siglip2-base fusion | 3,000 | φ 分布、方法排名 | MUST | TODO | 抗"单模型"质疑 |
| R021 | M1 | 预处理消融 | 保留 vs 剥除 URL/mention | 500 | φ 分布变化 | NICE | TODO | 抗 A3 |
| R022 | M3 | TCAV 概念敏感性曲线 | 概念强度递增 | 500 | 曲线单调性 | NICE | TODO | 附录 |
