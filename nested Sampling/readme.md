# 统计计算大作业企划书：后验复杂度对边际似然估计算法性能的影响研究

> 嵌套采样（Nested Sampling）与基于 MCMC 的边际似然估计的比较分析  
> 版本日期：2026‑06‑12（最终版）

---

## 1. 研究背景与核心问题

贝叶斯统计通过结合先验信息与样本信息进行推断，其核心公式为

\[
p(\beta | y) = \frac{p(y|\beta)\,p(\beta)}{p(y)}
\]

其中：

- \(p(\beta)\) —— 先验分布  
- \(p(y|\beta)\) —— 似然函数  
- \(p(\beta|y)\) —— 后验分布  
- \(p(y)\) —— 边际似然（也称为证据）

边际似然定义为高维积分：

\[
p(y) = \int p(y|\beta)\,p(\beta)\,d\beta
\]

在贝叶斯模型选择中，贝叶斯因子为

\[
BF_{12} = \frac{p(y|M_1)}{p(y|M_2)}
\]

因此，边际似然的准确计算具有根本重要性。然而，标准 MCMC 方法只能获得后验样本，无法直接计算该积分常数。本研究旨在系统比较 **嵌套采样** 与 **基于 MCMC 的边际似然估计** 两种途径，考察后验分布维度增加时各方法的精度、效率与稳定性。

---

## 2. 研究目标

- **精度差异**：嵌套采样与 MCMC‑based 方法在边际似然估计值上是否存在显著差异？
- **维度影响**：随着数据维度（参数个数）增加，两种方法的运行效率如何变化？
- **稳定性**：后验分布从低维单峰到高维稀疏结构时，算法重复运行的标准差如何变化？
- **适用场景**：总结嵌套采样适用于何种后验分布，MCMC‑based 方法适用于何种场景。

---

## 3. 数据集与模型

### 3.1 数据集（构造后验维度梯度）

| 数据集 | 维度等级 | 样本量 | 特征数（维度） | 后验复杂度特点 |
|--------|----------|--------|----------------|----------------|
| **Pima Indians Diabetes** | 低维 | 768 | 8 | 简单，近似单峰高维高斯 |
| **Credit Card Fraud (creditcard)** | 中维 | 284,807 | 30 | 中等，特征间存在相关性，后验略有偏斜 |
| **TCGA (癌症基因组图谱)** | 高维 | 约 500‑1000 | > 1000 | 高维稀疏结构，后验存在强相关性 |

> **数据存储路径**：  
> `D:\homeworkcode\统计计算\Marginal-Likelihood-calculation\data preprocessing`  
> 请在该目录下存放三个数据集的预处理文件（例如 `pima.csv`, `creditcard.csv`, `tcga.csv`），每个文件包含特征矩阵 \(X\) 和二分类标签向量 \(y\)。

### 3.2 贝叶斯逻辑回归模型（唯一模型）

#### 观测模型

\[
y_i \sim \text{Bernoulli}(p_i), \quad p_i = \frac{1}{1 + \exp(-x_i^T \beta)}
\]

#### 先验分布

高斯先验：\(\beta \sim N(0, 10I)\)

#### 对数似然

\[
\log p(y|\beta) = \sum_{i=1}^{n} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
\]

#### 对数先验

\[
\log p(\beta) = -\frac{1}{2} \left( \frac{\|\beta\|^2}{10} + p \log(2\pi \cdot 10) \right)
\]

#### 对数后验

\[
\log p(\beta|y) = \log p(y|\beta) + \log p(\beta)
\]

#### 后验梯度（用于 HMC）

\[
\nabla \log p(\beta|y) = X^T (y - p) - \frac{\beta}{10}
\]

> 编程实现函数：  
> `log_likelihood(beta)`, `log_prior(beta)`, `log_posterior(beta)`, `grad_log_posterior(beta)`

---

## 4. 两种边际似然估计方法

### 4.1 方法一：嵌套采样（Nested Sampling）

- **原理**：将多维积分 \( Z = \int \mathcal{L}(\theta) \pi(\theta) d\theta \) 转化为一维积分 \( Z = \int_0^1 \mathcal{L}(X) dX \)，其中 \( X \) 为先验体积。通过维护一组活点（live points）并逐步收缩似然阈值，估计边际似然。
- **实现库**：`dynesty`（动态嵌套采样）或 `UltraNest`（响应式嵌套采样）。本项目拟同时使用两者进行交叉验证。
- **输入**：`log_likelihood(beta)`, `log_prior(beta)`
- **输出**：`logZ`（对数边际似然估计），`logZ_error`（误差预算），`H`（KL 信息量），`runtime`，`ESS`（有效样本量，仅部分库输出）

### 4.2 方法二：基于 MCMC 的边际似然估计（MCMC‑based）

- **原理**：首先使用 MCMC（具体采用 **哈密顿蒙特卡洛 HMC**）从后验分布 \( p(\beta|y) \) 中采集大量样本。然后采用 **谐波均值估计（Harmonic Mean Estimator）** 计算边际似然：

\[
\hat{p}(y) = \left[ \frac{1}{N} \sum_{i=1}^{N} \frac{1}{p(y|\beta^{(i)})} \right]^{-1}
\]

  该估计量仅依赖于后验样本和似然函数，无需额外提议分布。虽然方差较大且可能不稳定，但作为与嵌套采样的对比基准具有教学意义。

- **实现步骤**：
  1. 运行 HMC 获得后验样本 \( \{\beta^{(1)}, ..., \beta^{(N)}\} \)。
  2. 计算每个样本的似然 \( p(y|\beta^{(i)}) \)。
  3. 按谐波均值公式计算 \( \log \hat{p}(y) \)。
- **输入**：`log_likelihood`, `log_prior`, `grad_log_posterior`（用于 HMC）
- **输出**：`logZ`（对数边际似然估计），`runtime`（包括 HMC 采样时间），`ESS`（HMC 的有效样本量）

> **注意**：谐波均值估计已知存在方差无穷大的问题，但在高维下仍可提供粗略对比。为了更稳健的对比，也可考虑 **拉普拉斯近似** 或 **重要性采样**，但为保持与“仅 MCMC”对应，本实验仍采用谐波均值。

---

## 5. 统一评价指标体系

所有实验均记录以下指标，每个配置独立运行 **30 次**。

| 指标 | 测量目标 | 计算/获取方式 |
|------|----------|----------------|
| **边际似然估计值 `logZ`** | 核心精度 | 嵌套采样：直接输出；MCMC：谐波均值公式 |
| **重复运行标准差 `std(logZ)`** | 算法稳定性 | 30 次运行结果的样本标准差 |
| **运行时间 `runtime` (秒)** | 计算效率 | 使用 `time.perf_counter()` 计时 |
| **有效样本量 `ESS`** | 后验采样质量 | MCMC：`pymc3`/`numpyro` 内置函数；嵌套采样：`dynesty` 可计算 `ess` |

### 5.1 嵌套采样特有的附加指标

#### KL 信息量 \( H \)
- **定义**：从先验到后验的信息增益，\( H = \mathbb{E}_{\text{posterior}}[\log(p(\beta|y)/p(\beta))] \)。
- **计算公式**：在嵌套采样中，\( H = \frac{1}{Z} \int \mathcal{L}(\theta) \log \mathcal{L}(\theta) \pi(\theta) d\theta - \log Z \)。
- **API 获取**：
  - `dynesty`：结果字典中的 `'information'` 键。
  - `UltraNest`：`results['nested_sampling_run']['H']`。

#### 误差预算（Error Budget）
- **定义**：由先验体积随机性导致的对数证据单次运行标准差。
- **API 获取**：
  - `dynesty`：使用 `utils.resample_run(results)` 生成模拟副本，计算 `lnZ` 的标准差。
  - 示例代码：
    ```python
    from dynesty import utils as dyfunc
    results_sim = dyfunc.resample_run(results, nsim=100)
    logz_error = results_sim['logz'].std()