# Sampling Strategy Memory
Date: 2026-04-20 (triple-confirmed null: H1+H4 × models × tiers × 11 metrics)

---

## TL;DR (latest finding, supersedes earlier "tentative" conclusions)

**Sampling strategy is dead as a contribution.** Confirmed across 8 strategies × 4 models on FWI 6-band probe — all PR-AUC within ±0.005, well within seed std. Linear vs nonlinear models give same null. Paper pivots to:
1. **Dataset** (full Canada × 25yr × 55ch)
2. **Benchmark + generalization framework** (TreeFinder-style multi-scenario)
3. **Interpretability via modality contribution decomposition** (absorbs SHAP)

Sampling demoted to Appendix "documentation experiment".

---

## 最终采样策略设计

### 删除的层
- ~~S2（生态区配额）~~：数据证明无严重欠采样，强制均匀违背真实分布
- ~~S4（空间去冗余）~~：全国尺度自然间距已接近10km阈值，边际效果极弱

### 保留的层

**L1 — 负样本日期来源**
- 从火季（5–10月）有火日期选负样本像素
- "有火日期"在加拿大尺度 ≈ 高CFFDRS日期（98.6%火季日有火，74%是高CFFDRS）
- 日期层面的CFFDRS筛选无实质区分度，直接用有火日期作代理

**L2 — 标签去污染**
- 仅对高CFFDRS日期的负样本施加30km/±3-day缓冲
- 缓冲基于firemask=9（高置信度火点）
- 注：firemask置信度对缓冲的影响已讨论但暂未在脚本中显式文档化

---

## 当前实验Variants（生产 A/B/C/D/E）

| Variant | 描述 | 文件路径 |
|---------|------|---------|
| A | Random 2:1 negatives, no filter layers | cache_A/samples_variant_A.json |
| B | + Layer 1 (buffer 30km/±3d) | cache_B/samples_variant_B.json |
| C | + Layer 2 (ecozone × LC × month stratification) — **production** | cache_C/samples_variant_C.json |
| D | + Layer 3 (50% hard-neg by composite FWI) | cache_D/samples_variant_D.json |
| E | + Layer 4 (10km spatial dedup) — full pipeline | cache_E/samples_variant_E.json |

所有variant：
- 固定测试集：50,000 pos + 100,000 neg（2023–2025年）
- Train: 2000–2019（pos=144,879 / neg=289,758），Val: 2020–2022（pos=22,548 / neg=45,096）

旧的Z/Y/G变体仅作legacy probe使用，已不在生产pipeline中。

---

## 关键数据分析结果

### CFFDRS面积分析
- FWI强度阈值（火点P25）：4.54
- 自然断点：0.058（5.8%研究区面积FWI≥4.54）
- 高CFFDRS日：3,520天（74%），低CFFDRS日：1,235天（26%）
- 火季总天数：4,755天，其中无火日仅66天（1.4%）
- **结论**：火季内有火日期≈高CFFDRS日期，日期层面筛选无实质区分度

---

## H1 + H4 探针：终结性零结果（2026-04-20）

### 实验设计
- 探针：5k pos + 10k neg train，5k pos + 10k neg val（2020-2022 OOD）
- 特征：FWI 6-band
- 候选池：变体B的train negatives（289k buffer-filtered随机）
- 评估：20 seeds × LR (单模型实验), 10 seeds × {LR, RF, GBM, MLP} (模型扫描)
- 候选策略（8种）：
  - **Random**：B池随机
  - **C_baseline**：C池随机（50%composite-hard + 50%random）
  - **H1_composite**：B池top-N by Σwᵢ × clip(FWIᵢ/p99ᵢ)
  - **H4_FFMC / DMC / DC / ISI / BUI**：B池top-N by单一分量

### 结果1：LR单模型 20 seeds
| Strategy | PR-AUC |
|----------|--------|
| Random | 0.8474 ± 0.0013 |
| C_baseline | **0.8467** ± 0.0013 |
| H1_composite | 0.8475 ± 0.0012 |
| H4_FFMC | 0.8476 ± 0.0011 |
| H4_DMC | 0.8474 ± 0.0012 |
| H4_DC | 0.8476 ± 0.0012 |
| H4_ISI | 0.8475 ± 0.0011 |
| H4_BUI | 0.8472 ± 0.0012 |

最大差距 0.0008，**等于typical std**——完全打不开。
C_baseline反而最低，因为hard-neg在FWI-only probe下相当于"故意挑最难的题"。

### 结果2：模型扫描（LR/RF/GBM/MLP × 10 seeds）
| Model | Random | C_baseline | H1_comp | H4_FFMC | H4_DMC | H4_DC | H4_ISI | H4_BUI | spread | std |
|-------|--------|-----------|---------|---------|---------|--------|--------|--------|--------|-----|
| LR | 0.8479 | 0.8471 | 0.8478 | 0.8476 | 0.8479 | 0.8479 | 0.8479 | 0.8472 | 0.0008 | 0.0010 |
| RF | 0.8158 | 0.8163 | 0.8157 | 0.8139 | 0.8160 | 0.8152 | 0.8156 | 0.8142 | 0.0024 | 0.0040 |
| GBM | 0.8446 | 0.8436 | 0.8440 | 0.8416 | 0.8459 | 0.8445 | 0.8445 | 0.8423 | 0.0043 | 0.0040 |
| MLP | 0.8451 | 0.8392 | 0.8461 | 0.8453 | 0.8457 | 0.8440 | 0.8434 | 0.8418 | 0.0069 | 0.0055 |

**每个模型的"赢家"都不同**（Random / C_baseline / H4_DMC / H1_composite）→ 完全是随机抖动。
即便最敏感的MLP，spread (0.007) 也几乎等于其std (0.006)。
**结论：模型复杂度不是confounder，sampling策略真正无差异。**

### 已被证伪的假设
- 日期级CFFDRS筛选（早期记忆已知）
- pixel-level composite hard-neg（H1）
- pixel-level单分量hard-neg（H4 × 5）
- 线性 vs 非线性模型敏感度差异（4模型一致打平）
- "更复杂模型能挖出细微差距"（被MLP/GBM否决）

### 已是OOD还打平的关键观察
所有probe的eval set用的是val 2020-2022，已经是temporal OOD（vs train 2000-2019）。
即"OOD环境下sampling才有差异"这个pivot假设也不成立。
唯一未测的spatial OOD按同逻辑大概率也flat。

---

## 论文贡献定位（2026-04-20确定）

### 三个contribution
1. **Dataset**（25yr × 全Canada × 55通道） — 强，scope是核心卖点
2. **Benchmark + 泛化能力评估框架** — TreeFinder式多场景protocol（见下）
3. **多模态贡献分解 / 可解释性** — 吸收SHAP计划，比per-feature SHAP更深更actionable

### 采样策略章节定位
- **Appendix中的documentation experiment**：写"我们系统测试了8种策略 × 4种模型，所有差距≤0.005 PR-AUC，证明全国尺度下采样策略不敏感，最终采用buffer-based简单方案"
- **不再作为novelty claim**

### 生成框架（TreeFinder-inspired）

参考：`D:\OneDrive - University of Calgary\个人文档\entire_canada_wildfire_dataset\neurips_references\TreeFinder A US-Scale Benchmark Dataset...md`

7个评估场景（每个分easy/hard两档）：
1. **Random temporal split**（baseline）+ learning curve（10%-80%）
2. **Cross-region**: Easy = West↔East Canada双向；Hard = BC → 其他所有省份
3. **Cross-ecozone (LOEO)**: 15个加拿大ecozone，每次留一个出来
4. **Cross-climate (Köppen)**: Easy = Dfb+Dfc → 其他；Hard = ET (tundra) → 其他
5. **Cross-LULC**: Easy = Top-3 LULC类 → 其他；Hard = 单一boreal coniferous → 其他
6. **Cross-temporal-era** ⭐ (TreeFinder没有的优势): Easy = 2000-2010 → 2015-2025；Hard = 2000-2005 → 2020-2025
7. **Cross-fire-regime** (optional): lightning-dominated regions → human-ignition regions

模型阵容（参考TreeFinder的6模型benchmark矩阵）：
- 时序baseline: U-Net 3D, ConvLSTM
- 时空Transformer: TimeSformer, Video Swin
- 专用时空预报: SimVP, PredRNN
- 简单baseline: random forest on tabular features
- 终极baseline: 25-year per-pixel climatology

### 实施计划
| Phase | 内容 | 估时 |
|-------|------|------|
| A | Ecozone/Köppen mask准备、split JSON生成 | 2-3天 |
| B | Pipeline改造支持任意train/test mask | 2-3天 |
| C | 跑 7 scenarios × 4-6 models | 1-2周（GPU） |
| D | Tables + figures + writeup | 1周 |

---

## 脚本位置

### 当前session新增（H1+H4 probe）
- `sampling/probe_h1_h4/probe_h1_h4.py` — LR单模型8策略probe（20 seeds）
- `sampling/probe_h1_h4/probe_h1_h4_models.py` — LR/RF/GBM/MLP 4模型扫描（10 seeds）
- `sampling/probe_h1_h4/results.json` — LR results
- `sampling/probe_h1_h4/results_models.json` — model sweep results
- `sampling/probe_h1_h4/{B_pos_tr,B_neg_tr,B_pos_va,B_neg_va,C_neg_tr}.parquet` — 缓存特征
- `/tmp/run_probe_models.sh` — setsid wrapper（python -u 失效时备用）

### 历史scripts
- `sampling/generate_samples.py` — 主pipeline (A/B/C/D/E)
- `sampling/layer1_filter.py` — Layer 1 buffer/cloud过滤
- `sampling/layer3_hard_negative_scorer.py` — composite + p99 score class
- `sampling/fwi/outputs/weight_vector_w1w6.json` — 25yr calibrated weights + p99
- `/tmp/learning_curve.py` — Z/Y/G learning curve（legacy）

### Python环境
**关键**：必须使用 `/home/zhengsen/.conda/envs/hackathon/bin/python3` (np 1.26 / pd 2.3 / sk 1.6 / GDAL ok)。
系统python3 (numpy 1.21) 和 pandas 不兼容，会立刻报ImportError。

---

## 待办
- [ ] Phase A: 生成ecozone-based splits (LOEO + W/E split + BC-vs-others)
- [ ] Phase A: 叠加Köppen气候带mask到research area
- [ ] Phase B: 改造train pipeline支持任意train/test region mask
- [ ] Phase C: 完整benchmark矩阵（7 scenarios × 4-6 models）
- [ ] 论文Section 3.3-4写作
- [ ] 把sampling变体对比 → 移到Appendix A
- [ ] L2缓冲firemask 7/8包含问题（已搁置，需在methods Appendix交代）

---

## Hard-Positive Evaluation 扩展 (2026-04-20，同日第二轮)

**假设**（见 `hard_positive_evaluation_plan.md`）：sampling 策略在 FM=9 test 上打平，可能因为 FM=9 对 FWI 是 ceiling。在 FM=7/8（low/nominal confidence）positives 上，hard-neg mining 是否终于显出差异？

**结论：再次证伪。Sampling 在所有 difficulty tier + 所有 metric 下都打平。**

### 实验1：三档 difficulty baseline（FM9/8/7，Random 训练）
- Val pool: FM=7 n=6,667；FM=8 n=19,996；两者来自 2020-2022 火季，research-area mask
- 结果（LR / GBM, 10 seeds, PR-AUC）：
  - FM=9: 0.849 / 0.849（复现 ceiling）
  - FM=8: 0.768 / 0.775（-8pt）
  - FM=7: 0.754 / 0.758（-9.5pt vs FM9）
- **Tier gap 真实存在**但 LR≈GBM，非线性救不了 → 是 FWI 信号层面限制
- 大部分 gap 发生在 FM9→FM8（-8pt），FM8→FM7 只差 1.4pt（非光滑曲线）

### 实验2：8-strategy × 3-tier PR-AUC 扫描（10 seeds）
- LR column spread (max−min) 在三档 tier 都 ≤ 0.0008（seed std ≈ 0.001）
- GBM column spread ≤ 0.0037（seed std ≈ 0.003-0.005）
- **Random 在 FM=7 LR 下排第一**（反而最优）——hard-neg 没有任何优势
- 没有任何一个策略 × tier 组合超过 seed noise

### 实验3：11-metric × 8-strategy × 3-tier 全扫描（10 seeds）
Metrics: pr_auc, roc_auc, brier, ece, hit@1/5/10, R@FPR=1%/0.1%, P@FPR=1%/0.1%

- LR 下所有 metric × tier spread / std ≤ 1.3×（完全在噪声内）
- GBM 下最高 spread/std ≈ 1.8×（hit@1 @ FM=7）— borderline，cherry-pick
- 66 次比较中 Random 赢 7 次（最多），H4_DMC 5 次，H1_composite 5 次 — 轮流当冠军 = 随机分布
- **Operational metric (Recall@FPR) 也救不了** —— FM9→FM7 tier gap 42-46% 存在，但 strategy 之间没有 gap

### 决定性证据
**tier-level gap 真实（FWI 对 hard fires 弱）；但 strategy-level gap 在所有 metric 上都是 0** 。即：sampling 是否用 hard-neg 挑 negatives，不影响分类器在任何难度、任何指标下的表现。这是 FWI 6D 近线性特征空间的本质约束，不是 test-set bias 带来的 artifact。

### 剩下一条未测的线（但不建议追）
GBM hit@1 @ FM=7 下 H1_composite 0.945 vs H4_DC 0.918（Δ=0.027, std≈0.012, t≈2.2）。66 次比较光凭 α=0.05 也会有 ~3 个假阳性。不值得 30-seed follow-up。

### 对论文定位的影响
1. Sampling-as-novelty 彻底封死，移出 candidate list（从"appendix documentation experiment"降级为"already-documented null result"）
2. **但 tier-level gap（FM9→FM7 = 9.5pt PR-AUC，42-46% Recall@FPR，35% Brier）是 paper 的潜在 punchline**:
   - "FWI 在 continental scale 上对 high-confidence fires 是 ceiling，对 low-confidence (small/early/edge) fires 能力骤降"
   - 核心 figure：3-tier PR curve + reliability diagram + Recall-at-FPR curve
   - 支持 Plan B3 (FWI universality + operational blind spot) 或纳入 generalization framework 作 difficulty axis

### 本轮新增文件
- `sampling/probe_hardpos/extract_hardpos_pool.py` — FM7/FM8 positive 池生成
- `sampling/probe_hardpos/extract_features.py` — FWI feature 提取
- `sampling/probe_hardpos/probe_hardpos.py` — 三档 difficulty baseline probe
- `sampling/probe_hardpos/probe_hardpos_extra.py` — 三档 × 11 metrics（仅 Random 训练）
- `sampling/probe_hardpos/probe_strategies_x_tiers.py` — 8 策略 × 3 tier（PR-AUC）
- `sampling/probe_hardpos/probe_strategies_x_tiers_extra.py` — 8 策略 × 3 tier × 11 metrics
- `sampling/probe_hardpos/{pool_fm7,pool_fm8}_val.json` — positive 池
- `sampling/probe_hardpos/{fm7,fm8}_val.parquet` — 缓存 FWI 特征
- `sampling/probe_hardpos/results*.json` — 所有 probe 结果
