# Hard Positive Evaluation — Plan & Context for Next Session
Date: 2026-04-20
Author: continuation-of-session memory (context budget filling up)

---

## 0. TL;DR — what to do next session

**核心发现**：现有test set的正样本只有firemask=9（高置信度大火），这是**FWI被设计来预测的那类火**。所以ML没有提升空间、sampling策略打平、跨域benchmark大概率也flat——全是同一个circular evaluation问题的症状。

**下一步**：跑quick probe验证"在firemask=7/8（小/早期/边缘火）上FWI的PR-AUC是否会显著掉下去"。如果掉，整个paper围绕"stratified-by-detection-confidence evaluation"重组，所有contribution（dataset / generalization / modality decomposition）都豁然开朗。

**预期结果**（如果假设成立）：
- firemask=9: PR-AUC ~0.85（已知，FWI ceiling）
- firemask=8: PR-AUC ~0.70（估计）
- firemask=7: PR-AUC ~0.55-0.65（估计）

如果firemask=7 PR-AUC仍然 ≥ 0.80，假设破产，需要换思路（见Section 8）。

---

## 1. 关键背景（必读）

### 现有test set设计（biased）
位置：`/s3/zhengsen/wildfire_dataset/h5_dataset/entire_can_v1/cache_{A,B,C,D,E}/samples_variant_*.json`

每个variant都共享相同的test split：
- 50,000 positive samples，全部 firemask = 9
- 100,000 negative samples，全部 firemask = 5
- 时间范围：2023-2025

**问题**：firemask=9是MODIS最高置信度的fire detection，对应：
- 大面积、高温度对比的活跃火
- 信号明确、易于检测
- **正是CFFDRS/FWI被设计来预测的那类火**

而firemask=7（低置信度）和=8（中置信度）一直被当成"buffer/exclusion zone"使用——`layer1_filter.py:55-57`明确写：
```python
FIREMASK_HIGH_CONF   = 9          # positive label
FIREMASK_NOM_CONF    = 8          # exclusion buffer (not positive)
FIREMASK_LOW_CONF    = 7          # exclusion buffer (not positive)
```

**它们从未参与过任何评估。**

### MODIS firemask值定义（重要参考）
| 值 | 含义 |
|---|------|
| 0 | not processed (missing) |
| 3 | water |
| 4 | cloud |
| 5 | non-fire land (negative candidate) |
| 6 | unknown |
| 7 | low-confidence fire |
| 8 | nominal-confidence fire |
| 9 | high-confidence fire |

数据源：`/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged/`
文件命名：`yyyy_mm_dd.tif`，单波段GeoTIFF。

### 此前session已穷尽的实验（不要重复）

**H1+H4 sampling probes (2026-04-20，本session)**
- 8种采样策略 × 4模型 (LR/RF/GBM/MLP) × 10-20 seeds
- 所有差距 ≤ 0.005 PR-AUC，统计上完全打平
- **Root cause**: 现在我们意识到——FWI对firemask=9是ceiling，sampling动不了
- 文件：`sampling/probe_h1_h4/probe_h1_h4.py`, `probe_h1_h4_models.py`
- 缓存特征：`sampling/probe_h1_h4/{B_pos_tr,B_neg_tr,B_pos_va,B_neg_va,C_neg_tr}.parquet`

**已证伪的假设**：
1. 日期级CFFDRS筛选无效
2. 像素级CFFDRS hard-neg无效（H1和H4×5都试过）
3. 模型复杂度提升不能挖出差异
4. OOD eval（val 2020-2022 vs train 2000-2019）也flat

---

## 2. 假设与立论

### H_hard：在firemask=7/8正样本上，FWI预测能力会显著下降

**理由**：
- firemask=7/8对应的火往往是：
  - 小面积（sub-pixel）
  - 早期阶段（fire刚起，FWI条件可能尚未达到极端）
  - 边缘检测（高云覆盖、植被遮挡）
  - 不寻常天气下的火（比如"低FWI但燃烧"）
- FWI是基于"持续干燥+热风"推导的index，对这些非典型火的预测能力天然弱
- 操作上，**fire managers最想提前抓住的恰恰是这些small/early fires**——晚抓到就成了firemask=9，已经救不回来了

### 如果H_hard成立，整个paper重组

**新contribution结构**：
1. **Dataset** (25yr × 全Canada × 55ch) — 不变
2. **Stratified-by-detection-confidence benchmark** — 新contribution，论文核心
   - 三档positive难度（firemask 7/8/9）
   - 揭示标准benchmark（仅firemask=9）的盲区
   - 真正的operational relevance在hard tier
3. **Generalization × difficulty matrix** — 现在generalization framework才有意义
   - Easy positives跨域可能flat（FWI universality）
   - Hard positives跨域应该崩（FWI不universal）
   - 这个gap本身=paper的punchline

### 如果H_hard不成立（firemask=7的AUC仍然>0.80）

需要换方向（见Section 8）。

---

## 3. Quick Probe 实验设计

### 实验目标
单一问题：用相同probe（FWI 6-band + LR），在不同firemask的positives上的PR-AUC是多少？

### 实验设置
- **训练集**：复用现有B variant的train（pos=firemask=9, neg=firemask=5），5k pos + 10k neg per seed，从cached features抽
- **三个独立val test set**：
  - **Val_9**: firemask=9 (复用现有B_pos_va.parquet) + firemask=5 negatives (复用B_neg_va.parquet)
  - **Val_8**: 新生成 firemask=8 positives + 同一批 firemask=5 negatives
  - **Val_7**: 新生成 firemask=7 positives + 同一批 firemask=5 negatives
- 每个val set: 5k pos + 10k neg
- **关键**：训练时用的是firemask=9 positives（同现状），测试时换不同firemask positives，模拟"真实部署时遇到不同难度的火"

### 候选模型
两个就够了，避免context爆炸：
- **LR** (FWI 6-band) — 主要probe
- **GBM** (FWI 6-band) — sanity check，用n_estimators=200, max_depth=3

10 seeds即可（不需要20）。

### 预期输出
| Train Pos | Test Pos | Model | Expected PR-AUC |
|-----------|----------|-------|-----------------|
| FM=9 | FM=9 | LR | ~0.847 (复现已知) |
| FM=9 | FM=8 | LR | ? (假设：~0.70) |
| FM=9 | FM=7 | LR | ? (假设：~0.60) |
| FM=9 | FM=9 | GBM | ~0.844 (已知) |
| FM=9 | FM=8 | GBM | ? |
| FM=9 | FM=7 | GBM | ? |

### 决策标准（重要）

| firemask=7的PR-AUC | 解读 | 行动 |
|--------------------|------|------|
| ≥ 0.80 | 假设破产，FWI对所有难度都work | 换方向（Section 8） |
| 0.65–0.80 | 假设部分成立，gap存在但不戏剧性 | 继续hard-positive方向，但需additional metrics（calibration等）放大gap |
| < 0.65 | **假设强成立** | 全力推进hard-positive paper结构，立刻进入Phase B（cross-domain × difficulty matrix） |

---

## 4. 实施步骤（脚本layout）

### Step 1: 生成firemask=7/8 positive池

新脚本：`sampling/probe_hardpos/extract_hardpos_pool.py`

逻辑（伪代码）：
```python
# 复用 layer1_filter.py 的 Layer1Filter 类
# 但允许 positive_value 参数 (默认9，可设为7或8)
# 也复用 research_area mask 限制

from layer1_filter import (Layer1Filter, MODIS_DIR, FIREMASK_LOW_CONF,
                            FIREMASK_NOM_CONF, _get_research_union)
from datetime import date, timedelta
import json

def extract_positives_at_firemask(year_range, firemask_value):
    """复用Layer1的tile扫描逻辑，把positive_value换成7或8"""
    # 主循环遍历日期，读MODIS tile
    # 找出 firemask == firemask_value 的pixels
    # 限制在 research_area_shpfile_merge.shp 内
    # 不需要施加buffer（这是buffer之外的eval set）
    # 输出 [{year, day_of_year, date, pixel_row, pixel_col, firms_value, ...}, ...]
    ...

# 跑两次，分别生成firemask=7和firemask=8的val pool
# 限制在 2020-2022 (val years)
# 限制在每个firemask 50k样本（够用）
```

输出文件：
- `sampling/probe_hardpos/pool_fm7_val.json`
- `sampling/probe_hardpos/pool_fm8_val.json`

### Step 2: Extract FWI features

复用现有 `probe_h1_h4.py` 中的 `extract_features()` 函数（已经有tile读取+pixel lookup逻辑）。
建议日期subsample到200个val日期，控制成本。

输出 parquet：
- `sampling/probe_hardpos/fm7_val.parquet`
- `sampling/probe_hardpos/fm8_val.parquet`

### Step 3: Run probe

新脚本：`sampling/probe_hardpos/probe_hardpos.py`

复用结构（基本copy `probe_h1_h4_models.py`）：
```python
# 训练: 从cached B_pos_tr (firemask=9) + B_neg_tr 抽 5k+10k
# 评估: 三个val set分别评 (FM9/FM8/FM7 positives + 同样FM5 negatives)
# 模型: LR + GBM
# 输出表格
```

### Step 4: 分析与决策

按Section 3的决策标准判断。

---

## 5. 关键技术细节（避免上次踩坑）

### Python环境（必须用）
```
/home/zhengsen/.conda/envs/hackathon/bin/python3
```
- numpy 1.26 / pandas 2.3 / sklearn 1.6 / GDAL ok
- 系统python3 (numpy 1.21) 不兼容会立刻报错

### 后台运行wrapper
```bash
# /tmp/run_probe_models.sh 已存在，可参考
# 关键：用 setsid，不要直接 nohup ... &
setsid /path/to/wrapper.sh > log 2>&1 < /dev/null &
```
直接 `nohup python ... &` 在ssh里会被kill，必须 setsid 或写 wrapper script。

### 文件传输
本地 → 服务器：
```bash
cat /c/home/script.py | ssh myserver "cat > /server/path/script.py"
```
注意：上次有一次cat传空文件的bug，**上传后立刻 wc -l 验证行数**！

### MODIS tile读取（性能注意）
- ~1-2s per tile
- 3000个日期全读 ≈ 1小时
- **必须做日期subsample**：参考 probe_h1_h4.py 里的 `N_TRAIN_DATES = 250`
- 对val来说，N_VAL_DATES = 150 已经足够

### Research area mask
所有positive/negative pixels必须先mask到 `research_area_shpfile_merge.shp`：
- 路径：`/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp`
- Layer1Filter 已自动处理（_get_research_union cache）

### FWI weight & p99（已固定）
- `sampling/fwi/outputs/weight_vector_w1w6.json`
- 25yr calibration on FFMC/DMC/DC/ISI/BUI/FWI
- p99: [97.92, 337.51, 895.46, 20.84, 340.66, 83.19]

---

## 6. 数据完整性提醒

### 样本数量预估
基于MODIS全Canada的统计经验（粗估）：
- firemask=9: ~大量（已知test set能抽50k）
- firemask=8: 应该比9更多（中等置信度比高置信度更普遍）
- firemask=7: 可能也很多但分布不同

如果firemask=7在val years (2020-2022) 内不足5k——**降低到3k并报告，不要硬要**。

### 时间分布
- 火季 (5-10月) 应该有大量firemask=7/8
- 非火季应该很少（避免误报数据增加噪声）
- 建议先限制在 fire-season only

### 空间分布
firemask=7/8可能在：
- 高云覆盖区（PNW海岸、东部沿海）更多
- 低植被区（tundra、prairie边缘）有特殊分布

值得**记录**这些样本的空间分布——后续cross-region分析时会用到。

---

## 7. 如果H_hard成立后的Phase B规划

完整paper所需实验矩阵：
```
positives × cross_domain × models
   3档     ×    7场景    ×  4模型  ≈ 84 experiments
```

Phase B（约2周）：
1. **每个firemask档**重新生成完整train/val/test splits（不只是val）
2. **Cross-region**: 在每档positives上跑West↔East
3. **Cross-ecozone (LOEO)**: 15 ecozones × 3 difficulty
4. **Cross-temporal-era**: 2000-2010 → 2015-2025，每档positives
5. **Modality decomposition**: 在firemask=7上做ablation（FWI / vegetation / weather / terrain），看哪些modality在hard tier上才显出价值

### Paper核心figure idea
**3 × 3 heatmap**: 行=训练集region, 列=测试集region, 颜色=PR-AUC drop relative to in-region baseline。
**做三张这样的figure**（每个firemask档一张），并排展示difficulty放大cross-domain gap的程度。

---

## 8. Plan B：如果firemask=7的PR-AUC仍然 ≥ 0.80

意味着FWI对所有难度都work，hard-positive假设破产。备选方向：

### Plan B1: 换eval metric
- **Calibration**: Brier score, Expected Calibration Error, reliability diagram
- **Top-K hit rate**: 火季top 10%大火的预测排名
- **Precision @ low FPR**: P@1%FPR, P@0.1%FPR
- **Lead time × accuracy curve**: 1d/7d/14d ahead

这些metric对FWI的universality更不友好。

### Plan B2: 评估在harder positive subset
- 大火（top 10% by burned area）
- 季节边缘火（早春/晚秋）
- 低FWI实际燃烧的火（"FWI surprise"）

### Plan B3: 拥抱null finding成为paper
重写为"FWI universality at continental scale"——RSE可能买账，NeurIPS不买。
需要做得无懈可击：
- 完整跨ecozone/climate/temporal validation
- 对比各种ML model都不能beat FWI
- Discussion明确说明implications for operational systems

### Plan B4: 完全改方向
- **Multi-horizon forecasting**: 1d/7d/14d/30d × cross-domain
- **Climate trend analysis**: 用25年数据正面研究fire weather drift
- **Modality cost-benefit**: 用sampling/feature ablation matrix做operational guide

---

## 9. 此session关键决策记录

1. ✅ Sampling策略不能作为novelty（已三重确认：单模型LR、模型扫描、OOD eval全flat）
2. ✅ Three-contribution新结构：dataset + benchmark/generalization + interpretability/modality
3. ✅ Generalization framework参考 TreeFinder 4-scenario protocol
4. ✅ 加上Cross-temporal-era作为TreeFinder没有的unique advantage
5. ✅ **NEW (this hour)**: 意识到问题根源在test set bias——只有firemask=9
6. ⏳ **Pending**: 验证hard positive假设（next session第一件事）

---

## 10. 关键文件locations速查

### 这次session新增（保留）
- `sampling/probe_h1_h4/probe_h1_h4.py` — H1+H4 LR probe
- `sampling/probe_h1_h4/probe_h1_h4_models.py` — 模型扫描
- `sampling/probe_h1_h4/results.json`, `results_models.json` — 结果
- `sampling/probe_h1_h4/{B_pos_tr,B_neg_tr,B_pos_va,B_neg_va,C_neg_tr}.parquet` — 缓存特征（**复用！**）
- `sampling/memory/sampling_strategy_memory.md` — 已更新的主记忆
- `sampling/memory/hard_positive_evaluation_plan.md` — **本文件**

### 下次session要新建
- `sampling/probe_hardpos/extract_hardpos_pool.py` — 生成FM7/FM8 positive池
- `sampling/probe_hardpos/probe_hardpos.py` — 三档difficulty probe
- `sampling/probe_hardpos/{fm7,fm8}_val.parquet` — FM7/8特征缓存
- `sampling/probe_hardpos/results.json` — 三档结果

### 关键现有scripts/data
- `sampling/layer1_filter.py` — Layer1Filter类，**复用firemask读取逻辑**
- `sampling/layer3_hard_negative_scorer.py` — FWI scoring class（备用）
- `sampling/fwi/outputs/weight_vector_w1w6.json` — calibrated weights + p99
- `/s3/zhengsen/wildfire_dataset/h5_dataset/entire_can_v1/cache_*/samples_variant_*.json` — 现有variants
- MODIS tiles: `/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged/yyyy_mm_dd.tif`
- FWI tiles: `/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/fwi_components/fwis_yyyy_mm_dd.tif`

### Paper
- `/mnt/raid/zhengsen/hackathons/dataset_building_entire_canada/paper/v0.tex`

---

## 11. 下次session第一句话怎么开始

> "继续hard positive evaluation。读 `sampling/memory/hard_positive_evaluation_plan.md`。先实施Step 1（extract_hardpos_pool.py）生成firemask=7/8 val positives，然后跑probe_hardpos.py，按Section 3的决策标准判断。"

如果时间紧，可以让我直接从Step 1开始，不需要重新讨论方向——这份记忆已经包含完整决策依据。

---

## 12. 小提醒

- 不要再回头测sampling策略（已穷尽）
- 不要测temporal OOD on firemask=9（已知flat）
- 不要直接跳到完整benchmark矩阵（**先验证hard positive假设**，不然3-4周compute白费）
- 写脚本时记得：MODIS要masked到research area，要做日期subsample，必须用 hackathon conda env
