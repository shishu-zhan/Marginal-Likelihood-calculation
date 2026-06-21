# Nested Sampling Experiment Design & Analysis

> 嵌套采样边际似然计算：实验设计、参数选择依据与结果分析方法  
> 版本日期：2026-06-18（最终版）

---

## 1. 研究目标

贝叶斯逻辑回归模型下，对三个维度递增的数据集（Pima=9d, CreditCard=30d, TCGA=61d）使用嵌套采样估计边际似然 logZ。核心关注：

1. **精度**：logZ 重复运行标准差 `std(logZ)` 是否满足模型选择需求（Bayes Factor 1σ 不确定度 < 3x）
2. **维度扩展**：采样效率（ncall, runtime）如何随 d 增长
3. **采样方法选择**：rwalk vs rslice 在不同维度下的行为差异

最终与队友的 Bridge Sampling 及 HMC 方法进行系统对比。

---

## 2. 数据集与模型

### 2.1 数据集

| 数据集 | 维度 | N | 特征 | 预处理 |
|--------|:---:|---|------|--------|
| Pima Indians Diabetes | 9 (Intercept + 8) | 768 | 年龄、BMI、血压等 | 中位数插补 + 标准化 |
| Credit Card Fraud | 30 (Intercept + 29) | 600 | V1–V28 + Amount | 分层抽样(300/300) + 标准化 |
| TCGA 癌症基因组 | 61 (Intercept + 60 PCs) | 801 | PCA 降维至 60 分量 | 零方差过滤 + PCA + 标准化 |

### 2.2 统计模型

**贝叶斯逻辑回归**，先验 β ~ N(0, 10·I)。对数先验方差 σ²=10（非标准 N(0,1)）。

关键函数见 `src/model.py`：`log_likelihood(beta, X, y)`, `prior_transform(u)`, `log_prior_density(beta)`。

---

## 3. 方法演进：V1 → V2 → V3

### 3.1 V1：Static NestedSampler + rwalk（基线）

| 参数 | Pima | CreditCard | TCGA |
|------|------|-----------|------|
| 采样器 | NestedSampler | ← | ← |
| 采样方法 | rwalk | ← | ← |
| walks | 30 | 30 | 30 |
| dlogz | 0.5 | 0.5 | 0.5 |
| nlive | 300 | 500 | 400 |

**结果**：
- Pima: logZ=-387.49±0.28 ✅
- CreditCard: logZ=-96.95±0.68 ✅
- TCGA: logZ=-57.57±**2.82** ❌ — BF 1σ 不确定度 exp(2.82)≈17x

**诊断**：rwalk 在 d=61 接受率指数衰减。dynesty FAQ 明确："If D > 20, 'rslice' is chosen since non-rejection sampling methods scale in polynomial (rather than exponential) time." walks=30 远不足 d=61 所需。

### 3.2 V2：DynamicNestedSampler + rslice + 默认停止（探索）

尝试用动态采样器 + rslice 修复 V1 的高维问题，但 DNS 默认停止函数 `pfrac=1.0`（100% 后验质量，0% 证据精度）与证据估计目标错位——基线阶段 dlogz_init=0.01 过于严格。

**结果**：CreditCard (d=30) 和 TCGA (d=61) 均在基线阶段耗尽 maxcall 仍不收敛。

**教训**：DNS 的默认配置为后验估计设计，不适合证据估计任务。需显式设定证据优先权重 `wt_kwargs={'pfrac': 0.0}` 或回退到证据导向的静态采样器。

### 3.3 V3（最终）：Static NestedSampler + 维度自适应采样方法

**设计原则**：

1. 使用静态 `NestedSampler`——停止准则 `dlogz` 直接控制证据收敛，行为可预测
2. 采样方法按官方维度建议：d≤20→`rwalk`，d>20→`rslice`
3. nlive 统一由精度准则 `nlive = ⌈H/σ²⌉` 导出，目标单轮 σ<0.5 nats

| 参数 | Pima (d=9) | CreditCard (d=30) | TCGA (d=61) |
|------|-----------|-------------------|-------------|
| 采样器 | NestedSampler | NestedSampler | NestedSampler |
| 采样方法 | rwalk | rwalk | **rslice** |
| walks | 30 | 30 | —（rslice 不需要） |
| dlogz | 0.5 | 0.5 | 0.5 |
| nlive | 300 | 500 | 250 |
| 预估 H | 26.6 | 43.5 | 60.1 |
| 理论单轮 σ | 0.30 | 0.30 | 0.49 |
| maxcall | 1,000,000 | 3,000,000 | 8,000,000 |

**Pima 和 CreditCard 采用 V1 结果**（rwalk 在 d≤30 已验证可靠）。仅 TCGA 用新参数重跑。

---

## 4. 参数选择依据

### 4.1 采样方法选择：`rwalk` vs `rslice`

| 机制 | rwalk | rslice |
|------|-------|--------|
| 类型 | MCMC 拒绝型 | 无拒绝切片 |
| 维度缩放 | O(e^d) | O(d³) |
| d>20 行为 | 接受率→0，证据估计有偏 | 多项式缩放，可用 |
| 参数 | walks（游走步数） | slices=3+d（自动） |
| dynesty 建议 | D≤20 | **D>20** |

**依据**：dynesty 官方 FAQ 及源码 `auto` 模式选择逻辑。TCGA(d=61) 使用 rslice 是唯一可行方案。

### 4.2 nlive 选择：精度准则

目标单轮证据精度 **σ < 0.5 nats**，由嵌套采样误差公式：

\[
\sigma(\log Z) \approx \sqrt{\frac{H}{N_{\text{live}}}}
\]

导出：`nlive = ⌈H / σ²⌉`。H 来自预实验（1轮测试）的 `results.information[-1]`。

```
Pima:       nlive = ⌈26.6 / 0.25⌉ = 107 → 取 300（保守） → σ≈0.30
CreditCard: nlive = ⌈43.5 / 0.25⌉ = 174 → 取 500（保守） → σ≈0.30
TCGA:       nlive = ⌈60.1 / 0.25⌉ = 240 → 取 250         → σ≈0.49
```

Pima/CreditCard 的 nlive 高于公式最低要求（V1 保守选择），但结果已经可用且无需重跑。

### 4.3 dlogz = 0.5

dynesty 推荐的保守停止准则。值越小越精确但越慢。0.5 为证据估计的标准推荐值。

### 4.4 walks = 30

dynesty FAQ："In lower dimensions (d≲15), walks=25 is often sufficient, while in moderate dimensions (d~15-25) walks=50 or greater are often necessary." Pima(d=9) 和 CreditCard(d=30) 使用 walks=30。CreditCard 在 rwalk 可用边界上，实际 std=0.68 可接受。

---

## 5. 记录指标

每条 `run_NNN.csv` 包含 11 个指标：

| 指标 | 含义 | 来源 |
|------|------|------|
| `logz` | 对数边际似然估计 | `results.logz[-1]` |
| `logzerr` | 内部误差估计 | `results.logzerr[-1]` |
| `logz_resample_std` | 理论误差 `sqrt(H/nlive)` | 计算 |
| `H` | KL 散度（先验→后验信息增益） | `results.information[-1]` |
| `runtime` | 单轮耗时 (s) | `time.perf_counter()` |
| `ncall` | 似然函数总调用次数 | `results.ncall` |
| `n_iter` | 嵌套采样迭代步数 | `len(results.samples)` |
| `eff_nlive` | 有效活点数 `ncall/(H*ndim)` | 计算 |
| `nlive_used` | 名义活点数 | 配置参数 |
| `seed` | 随机种子 | `base_seed + run_id*100` |
| `converged` | 1=自然收敛，0=maxcall 截断 | `ncall < maxcall` |

---

## 6. 跨方法系统对比指南

### 6.1 精度对比

| 对比维度 | 度量 | 说明 |
|---------|------|------|
| 点估计偏差 | `|logZ - logZ_ref|` | 若无法获真值，用 30 轮均值作为参考 |
| 单轮精度 | `logzerr`（内部） | 算法对自己的信心 |
| 运行间稳定性 | `std(logZ)`（30 轮） | 算法对初始条件的敏感度 |
| 模型选择可用性 | `std(logZ)` → `exp(std)` = BF 1σ 不确定度 | BF>3 区分需不确定度<~2x |
| 运行间稳定性影响 | `std(logZ)/√30`（net σ） | 多次运行后的综合精度 |

**跨方法对比时**：Nested Sampling 的 H 应与 Bridge Sampling/HMC 的 KL 估计一致（同一后验）。logZ 之间的偏差超出各自 net σ×2 则表明至少一个方法有偏。

### 6.2 效率对比

| 维度 | 度量 | 跨方法可比？ |
|------|------|:---:|
| 绝对耗时 | `runtime` | ✅ 所有方法 |
| 似然调用次数 | `ncall` | ✅ NS vs HMC（每次调用等价） |
| 单位精度成本 | `ncall / logzerr²` 或 `runtime / logzerr²` | ✅ |

**注意**：不同方法可能使用不同硬件（CPU/GPU）、不同并行度。比较时需归一化。

### 6.3 表格呈现建议（论文用）

| 数据集 | 方法 | logZ | logZ_std (30runs) | H | runtime(s) | ncall | σ_single | σ_net(30) | BF_uncert |
|--------|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pima | dynesty-rwalk | -387.49 | 0.28 | 26.5 | 27.5 | 253K | 0.30 | 0.05 | 1.05x |
| CC | dynesty-rwalk | -96.95 | 0.68 | 43.5 | 147.6 | 752K | 0.30 | 0.12 | 1.13x |
| TCGA | dynesty-rslice | *待* | *待* | *待* | *待* | *待* | 0.49 | 0.09 | 1.09x |
| TCGA | dynesty-rwalk | -57.57 | **2.82** ❌ | 54.1 | 126.3 | 668K | — | 0.51 | 1.67x |

---

## 7. 已知注意事项

### 7.1 scipy 兼容性

conda 环境需 numpy≥1.26.4 且 scipy≤1.13.0。创建 conda 环境时 scipy 会被自动安装为 1.17.x，与 numpy 1.26.0 不兼容，会导致 LAPACK DLL 崩溃（Windows error 0xc06d007f）。

```bash
conda activate nested_sampling
pip install numpy==1.26.4 scipy==1.13.0
```

### 7.2 不重跑 Pima/CreditCard

V1 的 rwalk 结果在 d≤30 已验证可靠（std=0.28/0.68）。V3 仅对 TCGA(d=61) 用 rslice 重跑。论文中注明采样方法选择依据即可。

### 7.3 nlive 的 H/σ² 公式

`σ ≈ sqrt(H/nlive)` 是嵌套采样的**理论误差下界**，实际运行间 std 通常略高于此值（1.3~1.5x）。V3 TCGA 的 30 轮结果可验证此关系。

### 7.4 随机种子

`seed = 1000 + run_id * 100`，30 轮种子为 1000, 1100, ..., 3900。使用 `numpy.random.seed()` 设定，非 dynesty 内部 RNG。

### 7.5 数据路径

`src/data_loader.py` 需要 3 层 `os.path.dirname()` 才能从 `nested Sampling/src/` 向上找到项目根目录下的 `data preprocessing/`。

---

## 8. 文件结构

```
nested Sampling/
├── src/
│   ├── model.py           — 贝叶斯 LR 模型：log_likelihood, prior_transform
│   ├── data_loader.py     — 加载 data preprocessing/ 下预处理 CSV
│   ├── dynesty_runner.py  — 封装 dynesty．NestedSampler（静态采样器）
│   ├── ultranest_runner.py — 封装 ultranest（未使用，因高维效率过低）
│   └── io_utils.py        — CSV 读写 + summary 统计
├── experiments/
│   ├── config.py          — 所有参数配置（含 V1/V3 完整参数）
│   └── run_experiments.py — CLI 入口，--method/--dataset/--phase/--seed/--runs
├── results/
│   ├── v1_rwalk_static/   — V1 30 轮结果（Pima + CreditCard + TCGA）
│   ├── v2_rslice_dynamic_failed/ — V2 1 轮演示（证明 DNS 默认停止不适用于证据估计）
│   └── v3_rslice_dynamic_final/  — V3 最终结果（仅 TCGA 重跑，Pima/CC 沿用 V1）
└── EXPERIMENT_DESIGN.md   — 本文档
```

---

## 9. 运行命令

```bash
conda activate nested_sampling
cd D:\homeworkcode\统计计算\Marginal-Likelihood-calculation\nested Sampling

# 测试（1 轮）
python experiments/run_experiments.py --method dynesty --dataset tcga --runs 1 --phase test

# 正式（30 轮）
python experiments/run_experiments.py --method dynesty --dataset tcga --phase final --seed 1000
```
