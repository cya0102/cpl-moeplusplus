# CPL_MoEv2：面向弱监督视频时序定位的尺度感知混合专家模型

## 摘要

CPL_MoEv2（Contrastive Proposal Learning with Scale-Aware Mixture of Experts v2）是在 CPL_MoE 基础上提出的改进模型，专门针对弱监督视频时序定位任务（Weakly-Supervised Video Temporal Grounding, WSVTG）中因长视频、长查询带来的性能退化问题。其核心创新在于将原始的朴素门控网络替换为**时序尺度感知门控系统（Scale-Aware Gating System）**，通过文本侧与视频侧的多尺度信号联合驱动专家路由，从而实现宏观（长时序）与微观（短时序）时序结构的自适应建模。本文对 `cpl_moev2.py` 中所有模块的设计动机、数学形式和实现细节进行完整阐述。

---

## 1. 背景与问题定义

### 1.1 任务设置

弱监督视频时序定位的目标是：给定一段自然语言查询 $q$（由 $L$ 个词组成）和一段视频 $V$（由 $T$ 帧组成），在无时序标注监督的条件下预测视频中与查询语义对应的时序区间 $[\hat{t}_s, \hat{t}_e]$。模型以**高斯提议参数**（中心 $\mu \in [0,1]$，宽度 $\sigma \in [0,1]$）隐式表示区间，由语义重建损失作为弱监督信号驱动学习。

### 1.2 CPL_MoE 的局限性

CPL_MoE 在 ActivityNet 数据集（长视频、长查询场景）上的召回率（R@1, IoU=0.3）约为 45.04%，较基线 CPL 的 55.73% 下降逾 10 个百分点。分析表明，其朴素门控网络在处理跨度为视频总长 30–50% 的动作片段时，对长时序与短时序动作的区分能力严重不足，导致专家分配出现"尺度错位"现象。CPL_MoEv2 正是为解决这一瓶颈而设计。

---

## 2. 模型总体架构

CPL_MoEv2 采用 **CPL 的双解码器 Transformer 骨干**（`DualTransformer`），在第一阶段跨模态编码之后插入时序尺度感知混合专家（MoE）系统，对视频时序特征进行尺度自适应增强，再送入提议预测头。

整体前向传播分为以下五个主要步骤：

$$
\underbrace{V, Q}_{\text{原始输入}} \xrightarrow{\text{投影+DualTransformer}} \underbrace{h_V, h_Q}_{\text{跨模态特征}} \xrightarrow{\text{Scale-Aware MoE}} \underbrace{\tilde{h}_V}_{\text{增强视频特征}} \xrightarrow{\text{提议头}} \underbrace{(\mu, \sigma)}_{\text{高斯参数}} \xrightarrow{\text{语义重建}} \mathcal{L}_\text{task}
$$

---

## 3. 模块一：尺度估计器（ScaleEstimator）

### 3.1 文本尺度编码器（QueryScaleEncoder）

`QueryScaleEncoder` 的设计目标是从投影后的词特征 $\mathbf{W} \in \mathbb{R}^{B \times L \times d}$ 中提取能够反映查询时序跨度的紧凑表示 $\mathbf{f}_q^{s} \in \mathbb{R}^{B \times d_s}$（其中 $d_s = d/4$）。

#### 3.1.1 注意力分布熵

首先对词特征施加一个轻量自注意力层（`nn.MultiheadAttention`，4 头），获得每头的注意力权重矩阵 $\mathbf{A} \in \mathbb{R}^{B \times H \times L \times L}$。

取第一个位置的注意力分布（类比 BERT 中 [CLS] token 的做法）：

$$
\mathbf{a}_h = \mathbf{A}_{:, h, 0, :} \in \mathbb{R}^{B \times L}
$$

对各头计算 Shannon 熵并求均值：

$$
\text{ent}_h = -\sum_{l=1}^{L} a_{h,l} \log(a_{h,l} + \epsilon), \quad \overline{H} = \frac{1}{H}\sum_{h=1}^{H} \text{ent}_h \in \mathbb{R}^{B}
$$

熵值高意味着注意力分布分散，对应查询中动作跨度较大（长时序动作）；熵值低意味着注意力集中，对应短时序、焦点明确的动作。

#### 3.1.2 时态关键词检测

为检测查询是否含有"持续时间相关"的表述（如"throughout"、"while"等），使用一个两层 MLP（`kw_detector`）对每个词位置输出一个 $[0,1]$ 的标量得分，并取有效位置内的最大值：

$$
\text{kw\_flag} = \max_{l \in \text{valid}} \sigma\!\left(\text{MLP}(w_l)\right) \in \mathbb{R}^{B}
$$

该检测器通过端到端训练学习识别与时序跨度相关的语义模式，规避了字符串匹配方法对词表的依赖。

#### 3.1.3 融合输出

将 $\overline{H}$ 与 $\text{kw\_flag}$ 拼接为二维信号，经线性映射至尺度特征空间：

$$
\mathbf{f}_q^{s} = \mathbf{W}_s \begin{bmatrix} \overline{H} \\ \text{kw\_flag} \end{bmatrix} \in \mathbb{R}^{B \times d_s}
$$

同时，对有效词进行 mask-pooling 得到全局查询表示：

$$
\mathbf{q}_\text{global} = \frac{\sum_{l} m_l \cdot \mathbf{w}_l}{\sum_{l} m_l + \epsilon} \in \mathbb{R}^{B \times d}
$$

### 3.2 视频密度分析器（VideoDensityAnalyzer）

`VideoDensityAnalyzer` 从视频特征 $\mathbf{V} \in \mathbb{R}^{B \times T \times d}$ 中提取多尺度时序变化节奏信息。

#### 3.2.1 多尺度时序差分能量

对步长 $k \in \{1, 4, 16\}$ 分别计算帧间差分：

$$
\Delta_k = \mathbf{V}_{:, k:, :} - \mathbf{V}_{:, :-k, :} \in \mathbb{R}^{B \times (T-k) \times d}
$$

计算 $L_2$ 范数得到逐时间步的变化能量，再分别提取均值和最大值：

$$
e_k^\text{mean} = \frac{1}{T-k}\sum_{t} \|\Delta_{k,t}\|_2, \quad e_k^\text{max} = \max_{t} \|\Delta_{k,t}\|_2
$$

将三个步长的 6 个统计量拼接，经线性映射得视频密度特征：

$$
\mathbf{f}_v^{s} = \mathbf{W}_d \left[e_1^\text{mean}, e_1^\text{max}, e_4^\text{mean}, e_4^\text{max}, e_{16}^\text{mean}, e_{16}^\text{max}\right]^\top \in \mathbb{R}^{B \times d_s}
$$

小步长（$k=1$）捕捉快速运动（局部动态），大步长（$k=16$）捕捉长程趋势（宏观动态），三者联合描述视频的时序节奏密度。

#### 3.2.2 多分辨率时序池化

同时计算两种分辨率的局部均值池化特征（窗口大小分别为 $T/8$ 和 $T/4$），取平均以获得涵盖局部结构的视频特征：

$$
\mathbf{v}_\text{local} = \frac{1}{2}\left(\text{AvgPool}_{T/8}(\mathbf{V}) + \text{AvgPool}_{T/4}(\mathbf{V})\right) \in \mathbb{R}^{B \times d}
$$

### 3.3 尺度得分计算

将文本侧尺度特征与视频侧密度特征拼接，通过线性层映射为标量：

$$
s = \sigma\!\left(\mathbf{w}_\text{scale}^\top \begin{bmatrix} \mathbf{f}_q^{s} \\ \mathbf{f}_v^{s} \end{bmatrix}\right) \in [0, 1]^{B}
$$

$s \to 1$ 表示当前样本倾向长时序动作（路由偏向宏观专家），$s \to 0$ 表示倾向短时序动作（路由偏向微观专家）。

---

## 4. 模块二：尺度感知门控网络（ScaleAwareGatingNetwork）

### 4.1 跨模态条件化

在计算路由逻辑之前，先用查询特征对视频全局特征进行跨模态条件化，消除模态间的语义偏移：

$$
\tilde{\mathbf{v}}_\text{cond} = \text{CrossAttn}\!\left(\mathbf{Q}=\mathbf{v}_\text{global}, \mathbf{K}=\mathbf{W}, \mathbf{V}=\mathbf{W}\right) \in \mathbb{R}^{B \times d}
$$

其中跨注意力模块（`LightCrossAttention`）采用 2 头注意力实现，以保持计算量轻量。

### 4.2 特征压缩

对各路特征分别施加 LayerNorm 后拼接，经线性层压缩至 $d$ 维：

$$
\mathbf{g} = \text{ReLU}\!\left(\mathbf{W}_c \left[\text{LN}(\tilde{\mathbf{v}}_\text{cond}) \,\|\, \text{LN}(\mathbf{q}_\text{global}) \,\|\, \text{LN}(\mathbf{f}_q^{s}) \,\|\, \text{LN}(\mathbf{f}_v^{s})\right]\right) \in \mathbb{R}^{B \times d}
$$

### 4.3 门控 MLP

两层 MLP 将压缩特征映射为 $N$ 个专家的原始 logit：

$$
\boldsymbol{\ell} = \mathbf{W}_2 \,\text{ReLU}\!\left(\mathbf{W}_1 \mathbf{g}\right) \in \mathbb{R}^{B \times N}
$$

### 4.4 尺度先验偏置（Scale Prior Bias）

基于尺度得分 $s$ 构造先验偏置，分别对前半（宏观专家组）和后半（微观专家组）施加方向相反的偏移：

$$
b_{i} = \begin{cases} +\alpha \cdot s & i < N/2 \quad \text{（宏观组）} \\ -\alpha \cdot s & i \geq N/2 \quad \text{（微观组）} \end{cases}
$$

其中 $\alpha$ 为可学习标量参数，从 0 线性预热至目标值 $\alpha_\text{target}$（详见第 7 节）。最终 logit 为：

$$
\boldsymbol{\ell}' = \boldsymbol{\ell} + \mathbf{b}
$$

此偏置在不强制硬分配的前提下，将尺度信号软性注入路由决策，规避了无路由标注的弱监督困境。

### 4.5 训练噪声退火

训练阶段在 logit 上叠加零均值高斯噪声以增强探索性：

$$
\boldsymbol{\ell}'' = \boldsymbol{\ell}' + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2 \mathbf{I})
$$

噪声标准差 $\sigma_n$ 从初始值 $\sigma_n^{(0)} = 0.5$ 线性退火至 0：

$$
\sigma_n^{(t)} = \sigma_n^{(0)} \cdot \left(1 - \frac{t}{T_\text{anneal}}\right)
$$

最终门控权重通过 Softmax 归一化：

$$
\mathbf{w} = \text{Softmax}(\boldsymbol{\ell}'') \in \Delta^{N-1}
$$

推理阶段关闭噪声，并采用 Top-2 稀疏路由（取权重最大的两个专家，归一化后加权聚合），兼顾精度与计算效率。

---

## 5. 模块三：分层专家池（StratifiedExpertPool）

专家池包含 $N$ 个专家，**前半（$[0, N/2)$）为宏观专家（MacroExpert），后半（$[N/2, N)$）为微观专家（MicroExpert）**，两组专家具有不同的结构性归纳偏置。

### 5.1 宏观专家（MacroExpert）——扩张自注意力

每个宏观专家使用**扩张采样的 Key/Value**，交替使用扩张步长 $d \in \{2, 4\}$：

$$
\text{KV}(x) = x[:, \{0, d, 2d, \ldots\}, :]
$$

通过稀疏采样的 K/V 对完整 Q 进行注意力计算：

$$
\text{out} = \text{Attn}(\mathbf{Q}=x, \mathbf{K}=\text{KV}(x), \mathbf{V}=\text{KV}(x))
$$

扩张机制使专家在不增加序列长度的情况下获得更宽的时序感受野，适合建模长程时序依赖。每个专家标准结构为：

$$
\text{MacroExpert}: \text{MHA}_\text{dilated} \to \text{Dropout} \to \text{LN} \to \text{FFN}(d \to 2d \to d) \to \text{Dropout} \to \text{LN}
$$

### 5.2 微观专家（MicroExpert）——滑动窗口自注意力

每个微观专家通过构造加法注意力掩码 $\mathbf{M}_\text{win}$（仅允许半窗口大小 $w/2$ 范围内的位置相互注意）实现**滑动窗口局部注意力**：

$$
M_{ij} = \begin{cases} 0 & |i - j| \leq w/2 \\ -\infty & |i - j| > w/2 \end{cases}, \quad w = T/4
$$

同时，使用 1D 卷积（`Conv1d`, $k=3$, padding=1）对 Value 进行局部投影，强化局部感受野：

$$
V' = \text{Conv1d}(x^\top)^\top
$$

完整结构为：

$$
\text{MicroExpert}: \text{MHA}_\text{window}(\mathbf{Q}=x, \mathbf{K}=x, \mathbf{V}=V') \to \text{Dropout} \to \text{LN} \to \text{FFN} \to \text{Dropout} \to \text{LN}
$$

### 5.3 位置编码保护机制

专家网络的自注意力操作可能破坏视频帧的绝对时序位置信息。为此，在每个专家输出后强制重新注入原始正弦位置编码 $\mathbf{PE}$：

$$
\tilde{o}_i = o_i + \mathbf{PE}, \quad \forall i \in [N]
$$

### 5.4 加权聚合

将 $N$ 个专家输出按门控权重加权求和：

$$
\mathbf{o}_\text{MoE} = \sum_{i=1}^{N} w_i \cdot \tilde{o}_i = \left(\mathbf{w}^\top \otimes \mathbf{I}_{T \times d}\right) \cdot \text{Stack}(\tilde{o}_1, \ldots, \tilde{o}_N)
$$

---

## 6. 模块四：残差连接与骨干适配

MoE 系统的输出通过残差连接与 LayerNorm 融入骨干特征，避免引入过多的特征漂移：

$$
\tilde{h}_V = \text{LN}\!\left(h_V + \mathbf{o}_\text{MoE}\right) \in \mathbb{R}^{B \times T \times d}
$$

增强后的视频特征 $\tilde{h}_V$ 代替原始视频特征进入下游的提议生成与语义重建阶段。

---

## 7. 提议生成

将 MoE 增强后的视频特征与骨干预测 token 输出（`pred_h`，对应帧序列末尾的特殊 token）融合，经池化后用高斯提议头预测每个提议的中心和宽度：

$$
\mathbf{f}_\text{mm} = \mathbf{h}_\text{pred} + \frac{\sum_t m_t \cdot \mathbf{o}_{\text{MoE},t}}{\sum_t m_t + \epsilon}
$$

$$
(\mu_p, \sigma_p) = \sigma\!\left(\mathbf{W}_g \mathbf{f}_\text{mm}\right), \quad p = 1, \ldots, N_\text{props}
$$

---

## 8. 损失函数

总损失由任务主损失与三个辅助正则化损失组成：

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{task} + \lambda_\text{bal} \mathcal{L}_\text{balance} + \lambda_\text{div} \mathcal{L}_\text{diversity} + \lambda_\text{ent} \mathcal{L}_\text{entropy}
$$

默认权重：$\lambda_\text{bal} = 0.1$，$\lambda_\text{div} = 0.05$，$\lambda_\text{ent} = 0.01$。

### 8.1 任务主损失（$\mathcal{L}_\text{task}$）

与 CPL_MoE 完全一致，包含：
- **重建损失（rec_loss）**：各提议的语义重建 NLL 损失取最小值（多实例弱监督），
- **对比排序损失（ivc_loss）**：正提议与负提议（时序偏移挖掘）之间的 max-margin 对比损失，以及多提议多样性约束（Gaussian 内积矩阵与对角目标的 Frobenius 距离）。

### 8.2 宏微观负载均衡损失（$\mathcal{L}_\text{balance}$）

防止所有样本全部流向同一专家组。对于每个样本，计算宏观组承载的权重之和，并对长查询样本加强约束力度：

$$
w_i = 1 + \beta \cdot \mathbb{1}[l_i > \bar{l}], \quad \bar{l} = \frac{1}{B}\sum_i l_i
$$

$$
\mathcal{L}_\text{balance} = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot \left(\sum_{j < N/2} w_{ij} - r_\text{macro}\right)^2
$$

其中 $r_\text{macro} = 0.6$ 为宏观组的目标承载比例，$\beta = 0.5$ 为长查询惩罚系数。

### 8.3 组间多样性损失（$\mathcal{L}_\text{diversity}$）

鼓励宏观组与微观组的路由分布在 batch 内存在显著差异，以最大化两组的 Jensen-Shannon 散度：

$$
\bar{p}_\text{mac} = \frac{1}{B}\sum_i w_{i,:N/2}, \quad \bar{p}_\text{mic} = \frac{1}{B}\sum_i w_{i,N/2:}
$$

$$
\mathbf{m} = \frac{1}{2}(\bar{p}_\text{mac} + \bar{p}_\text{mic}), \quad \text{JS}(p_1 \| p_2) = \frac{1}{2}\text{KL}(p_1 \| \mathbf{m}) + \frac{1}{2}\text{KL}(p_2 \| \mathbf{m})
$$

$$
\mathcal{L}_\text{diversity} = -\text{JS}(\bar{p}_\text{mac} \| \bar{p}_\text{mic})
$$

### 8.4 路由熵正则（$\mathcal{L}_\text{entropy}$）

防止路由退化为近 one-hot 分布（即专家塌缩），鼓励适度分散的软聚合：

$$
\mathcal{L}_\text{entropy} = -\frac{1}{B}\sum_{i=1}^{B} H(\mathbf{w}_i), \quad H(\mathbf{w}) = -\sum_{j=1}^{N} w_j \log(w_j + \epsilon)
$$

---

## 9. 两阶段训练策略

### 9.1 阶段一：骨干预热（Stage 1, Warm-up）

前 $E_\text{warmup}$（默认 5）个 epoch 冻结门控网络的全部参数，`gate_weights` 固定为均匀分布 $\mathbf{1}/N$。此阶段让各专家在均等权重下充分训练，获得基本的时序建模能力，为后续的差异化路由奠定基础。

### 9.2 阶段二：联合训练（Stage 2, Joint Training）

第 $E_\text{warmup}+1$ 个 epoch 起解冻门控网络，启用所有辅助损失，并同步执行：
- **噪声退火**：$\sigma_n$ 在 $T_\text{anneal}$ 步内线性从 0.5 衰减至 0，保证后期路由的稳定性；
- **$\alpha$ 预热**：Scale Prior Bias 强度 $\alpha$ 在 $T_\text{alpha}$ 步内线性从 0 预热至 $\alpha_\text{target} = 1.0$，避免早期训练中过强的先验干扰。

---

## 10. 消融实验开关

模型通过配置文件中的 7 个布尔开关支持逐模块消融，便于隔离各组件的贡献：

| 开关                     | 对应组件                       | 关闭时退化行为                      |
| ------------------------ | ------------------------------ | ----------------------------------- |
| `use_scale_estimator`    | ScaleEstimator                 | $s \equiv 0.5$，无尺度信号          |
| `use_stratified_routing` | MacroExpert/MicroExpert        | 仅保留结构（仍分层，无偏置）        |
| `use_scale_prior_bias`   | Scale Prior Bias               | $\mathbf{b} \equiv 0$，去除先验偏置 |
| `use_cross_attn_fusion`  | LightCrossAttention            | 跨模态条件化退化为恒等映射          |
| `use_balance_loss`       | $\mathcal{L}_\text{balance}$   | 关闭负载均衡                        |
| `use_diversity_loss`     | $\mathcal{L}_\text{diversity}$ | 关闭多样性正则                      |
| `use_entropy_loss`       | $\mathcal{L}_\text{entropy}$   | 关闭熵正则                          |

---

## 11. 路由可视化接口

`get_routing_stats()` 方法提供离线路由分析接口，返回每个样本的专家分配权重 $\mathbf{w}$、尺度得分 $s$ 以及宏观组总权重 $r_\text{macro}$，可用于按时序区间长度分组统计路由行为（如按 `gt_coverage_bucket` 划分），量化模型是否学到了有意义的尺度区分能力。

---

## 12. 与 CPL / CPL_MoE 的接口兼容性

CPL_MoEv2 保持与 CPL / CPL_MoE 完全相同的 `forward` 调用签名：

```python
def forward(self, frames_feat, frames_len, words_id, words_feat,
            words_len, weights, **kwargs) -> dict
```

返回字典中包含所有 `rec_loss` 和 `ivc_loss` 所需字段（`words_logit`、`words_id`、`gauss_weight`、`width`、`center` 等），因此**无需修改数据集、损失函数和评估流程**，可作为现有 `MainRunner` 的无缝替换。

---

## 13. 超参数汇总

| 超参数                      | 符号                   | 默认值      | 作用                           |
| --------------------------- | ---------------------- | ----------- | ------------------------------ |
| `num_experts`               | $N$                    | 4           | 专家总数（前半宏观，后半微观） |
| `d_scale`                   | $d_s$                  | 64（$d/4$） | 尺度特征维度                   |
| `lambda_balance`            | $\lambda_\text{bal}$   | 0.1         | 负载均衡损失权重               |
| `lambda_diversity`          | $\lambda_\text{div}$   | 0.05        | 多样性损失权重                 |
| `lambda_entropy`            | $\lambda_\text{ent}$   | 0.01        | 熵正则权重                     |
| `beta`                      | $\beta$                | 0.5         | 长查询重要性加权系数           |
| `target_macro_ratio`        | $r_\text{macro}$       | 0.6         | 宏观组目标负载比例             |
| `noise_init_std`            | $\sigma_n^{(0)}$       | 0.5         | 路由噪声初始标准差             |
| `alpha_target`              | $\alpha_\text{target}$ | 1.0         | Scale Prior Bias 目标强度      |
| `moev2_warmup_epochs`       | $E_\text{warmup}$      | 5           | 门控网络冻结轮数               |
| `moev2_noise_anneal_epochs` | $T_\text{anneal}$      | 15          | 噪声退火总 epoch 数            |
| `moev2_alpha_warmup_epochs` | $T_\alpha$             | 10          | $\alpha$ 预热 epoch 数         |
