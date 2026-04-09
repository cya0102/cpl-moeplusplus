# CPL_MoEv2 实现规格说明书

## 任务

基于现有的 `cpl_moe.py`，编写全新文件 `cpl_moev2.py`，实现改进模型 `CPL_MoEv2`。**不得修改原有的 `cpl_moe.py`**。

---

## 任务背景

1. **ActivityNet 瓶颈**：CPL_MoE 在 ActivityNet 上的整体召回率（R@1, IoU=0.3）仅为 **45.04%**，远低于 Baseline 的 **55.73%**。尤其在gt_coverage_bucket（真实标记的时序区间长度/视频总长度）为 **30-50%** 的切片中，性能落后达 **31.7%**。
2. **Charades 优势**：CPL_MoE 在短视频数据集 Charades 上表现稳健（**64.34%**），接近 Baseline（**66.05%**）。
3. **核心问题**：现有的门控网络（Gating Network）无法识别弱监督环境下的潜在动作尺度，导致在处理长查询（Long Query，缺口 **17.3%**）和长视频时，专家分配出现了“尺度错位”。

## 整体架构概述

`CPL_MoEv2` 在 `CPL_MoE` 的基础上，将原有的朴素门控网络替换为"时序尺度感知门控系统（Scale-Aware Gating System）"。该系统由以下五个核心模块组成，在 `forward` 中按顺序调用：

```
输入: video_feat [B, T, d], query_feat [B, L, d], query_mask [B, L]
  │
  ├─► [模块1] ScaleEstimator
  │     ├─ QueryScaleEncoder(query_feat, query_mask)  → query_scale_feat [B, d_scale]
  │     ├─ VideoDensityAnalyzer(video_feat)           → video_density_feat [B, d_scale]
  │     └─ 输出: scale_score (标量, [0,1]), query_scale_feat, video_density_feat
  │
  ├─► [模块2] ScaleAwareGatingNetwork
  │     ├─ 输入融合 (含跨模态 Cross-Attention + LayerNorm + 线性压缩)
  │     ├─ 计算 Scale Prior Bias
  │     └─ 输出: gate_weights [B, N] (softmax后)
  │
  ├─► [模块3-A] StratifiedExpertPool (MacroExperts + MicroExperts)
  │     ├─ 保存 positional_encoding pe [B, T, d]
  │     ├─ 各专家计算
  │     └─ 注入 pe: expert_out = expert_out + pe
  │
  ├─► [模块3-B] MoE 加权聚合 → moe_out [B, T, d]
  │
  ├─► [模块4] 残差连接 + LayerNorm
  │     h' = LayerNorm(h_backbone_L + moe_out)  → 送入骨干第 L+1 层
  │
  └─► 输出: 定位预测 + loss_dict (含所有损失项)
```

---

## 模块1：ScaleEstimator（尺度感知估计器）

### 1.1 QueryScaleEncoder（文本侧）

**目标**：提取查询的语义时序跨度特征，替代原方案中不可靠的词数代理。

**输入**：
- `query_feat`: `[B, L, d]`，文本编码器（BERT 等）的 token-level 输出
- `query_mask`: `[B, L]`，0/1 掩码，1 表示有效 token

**实现步骤**：

1. **注意力分布熵特征**：
   - 取文本编码器最后一层的 self-attention 权重矩阵 `attn_weights`: `[B, heads, L, L]`（需从编码器中 hook 或显式传入）。
   - 对每个 head 计算 CLS token 对其余 token 的注意力分布熵：
     ```
     H_head = -sum(p * log(p + eps), dim=-1)   # [B, heads]
     attn_entropy = mean(H_head, dim=-1)         # [B]，标量熵均值
     ```
   - 高熵表示查询关注分散（倾向长动作），低熵表示关注集中（倾向短动作）。

2. **时态关键词特征**：
   - 维护一个固定词表（duration adverbs），包含：`{"throughout", "while", "until", "during", "across", "entire", "whole", "from", "to", "continuously", "gradually"}`，共约 15 个词。
   - 对 `query_mask` 有效范围内的 token，检测是否包含上述关键词，生成二值向量 `kw_flag [B]`（1 = 含时态词）。
   - 将 `kw_flag` 转为 float 参与后续融合。

3. **融合输出**：
   - 全局文本特征：`q_global = mean_pooling(query_feat, query_mask)` → `[B, d]`
   - 将 `attn_entropy [B]` 和 `kw_flag [B]` 拼接为 `scale_signal [B, 2]`，经 `Linear(2, d_scale)` 映射为 `query_scale_feat [B, d_scale]`

**超参数**：`d_scale = d // 4`（默认值，可由 config 传入）

---

### 1.2 VideoDensityAnalyzer（视频侧）

**目标**：提取视频的多尺度时序变化节奏特征，替代原方案中语义不稳定的时序方差。

**输入**：`video_feat [B, T, d]`

**实现步骤**：

1. **多尺度时序差分能量（Multi-scale Temporal Difference Energy）**：
   - 对步长 `k ∈ {1, 4, 16}` 分别计算帧间差分：
     ```python
     # 对每个步长 k：
     diff_k = video_feat[:, k:, :] - video_feat[:, :-k, :]   # [B, T-k, d]
     energy_k = torch.norm(diff_k, dim=-1)                    # [B, T-k]
     mean_k = energy_k.mean(dim=-1)                           # [B]
     max_k  = energy_k.max(dim=-1).values                     # [B]
     ```
   - 拼接三个步长的均值和最大值，得 `density_raw [B, 6]`。
   - 经 `Linear(6, d_scale)` 映射为 `video_density_feat [B, d_scale]`。

2. **多分辨率池化特征**（用于门控输入）：
   - 全局特征：`v_global = video_feat.mean(dim=1)` → `[B, d]`
   - 局部特征（两个窗口尺度 `w ∈ {T//8, T//4}`，均向下取整，最小为 1）：
     ```python
     # 对每个窗口 w：使用 F.avg_pool1d，padding='valid'
     v_local_w = F.avg_pool1d(video_feat.transpose(1,2), kernel_size=w, stride=w//2)
                  .mean(dim=-1)   # [B, d]
     ```
   - `v_local = mean(v_local_w1, v_local_w2)` → `[B, d]`

**输出**：`video_density_feat [B, d_scale]`，`v_global [B, d]`，`v_local [B, d]`

---

### 1.3 ScaleScore 计算

将文本侧和视频侧信号融合，得到一个连续的尺度得分 `scale_score ∈ [0, 1]`：

```python
# query_scale_feat [B, d_scale], video_density_feat [B, d_scale]
fused = torch.cat([query_scale_feat, video_density_feat], dim=-1)  # [B, 2*d_scale]
scale_score = torch.sigmoid(self.scale_proj(fused))                 # [B, 1]，Linear(2*d_scale, 1)
scale_score = scale_score.squeeze(-1)                               # [B]
```

`scale_score` 接近 1 表示当前样本倾向于长时序（路由偏向 Macro-Experts），接近 0 表示倾向短时序（路由偏向 Micro-Experts）。

---

## 模块2：ScaleAwareGatingNetwork（尺度感知门控网络）

### 2.1 跨模态条件化（Cross-Modal Conditioning）

在拼接特征送入门控 MLP 之前，先用查询条件化视频全局特征，消除模态间语义偏移：

```python
# Q = v_global [B, 1, d]，K = V = query_feat [B, L, d]
v_conditioned = CrossAttention(Q=v_global.unsqueeze(1),
                               K=query_feat, V=query_feat,
                               key_padding_mask=~query_mask.bool())
# v_conditioned: [B, 1, d] → squeeze → [B, d]
v_conditioned = v_conditioned.squeeze(1)
```

**CrossAttention** 使用单头或 2 头注意力（轻量），`d_model = d`，无需 FFN。

### 2.2 特征拼接与压缩

```python
# 各路特征先 LayerNorm，再拼接
g1 = LayerNorm(v_conditioned)       # [B, d]   条件化后的视频全局特征
g2 = LayerNorm(q_global)            # [B, d]   查询全局特征（mean pooling）
g3 = LayerNorm(query_scale_feat)    # [B, d_scale]  文本尺度特征
g4 = LayerNorm(video_density_feat)  # [B, d_scale]  视频密度特征

gate_input_raw = torch.cat([g1, g2, g3, g4], dim=-1)   # [B, 2d + 2*d_scale]

# 线性压缩，防止维度爆炸
gate_input = Linear(2d + 2*d_scale, d)(gate_input_raw)  # [B, d]
gate_input = F.relu(gate_input)
```

### 2.3 门控 MLP 与 Scale Prior Bias

```python
# 门控 MLP 输出原始 logits
logits = gate_mlp(gate_input)   # [B, N]，gate_mlp: Linear(d, d//2) → ReLU → Linear(d//2, N)

# 构造 Scale Prior Bias
# scale_score [B]，alpha 为可学习标量（nn.Parameter，初始化为 1.0）
bias = torch.zeros(B, N, device=device)
bias[:, :N//2] = +self.alpha * scale_score.unsqueeze(-1)   # Macro 组
bias[:, N//2:] = -self.alpha * scale_score.unsqueeze(-1)   # Micro 组

# 训练初期 alpha 需预热（见模块6），推理时使用固定 alpha
logits = logits + bias

# 训练时：加入噪声退火（见模块6）
if self.training:
    noise = torch.randn_like(logits) * self.noise_std   # noise_std 随步数退火
    logits = logits + noise

gate_weights = F.softmax(logits, dim=-1)   # [B, N]
```

---

## 模块3：StratifiedExpertPool（分层专家池）

### 3.1 专家结构定义

共 `N` 个专家，**必须**给两组施加不同的结构性归纳偏置：

**Macro-Experts（索引 `[0, N//2)`）——长程注意力**：
```python
class MacroExpert(nn.Module):
    # Dilated Self-Attention，步长 dilation ∈ {2, 4} 交替
    # 实现方式：对 video_feat 按 dilation 步长采样后做 self-attention，输出插值回原长度 T
    # 等价简化实现：使用标准 nn.MultiheadAttention 但对 key/value 每隔 dilation 帧采样
    def __init__(self, d_model, dilation=2, nhead=4): ...
    def forward(self, x, pe): ...  # x: [B, T, d]
```

**Micro-Experts（索引 `[N//2, N)`）——局部滑动窗口注意力**：
```python
class MicroExpert(nn.Module):
    # 滑动窗口 self-attention，window_size = T // 4（动态，forward 时传入 T）
    # 使用 1D 卷积（kernel_size=3）作为 Value projection，强化局部感受野
    def __init__(self, d_model, nhead=4): ...
    def forward(self, x, pe): ...  # x: [B, T, d]
```

专家内部均包含：`MultiheadAttention → Dropout → LayerNorm → FFN(Linear-ReLU-Linear) → Dropout → LayerNorm`

### 3.2 位置编码保护机制

```python
def forward_expert_pool(self, video_feat, gate_weights, pe):
    # pe: 原始位置编码 [B, T, d]，在进入专家前保存

    expert_outputs = []
    for i, expert in enumerate(self.experts):
        out_i = expert(video_feat, pe)   # 专家内部可消费 pe
        # 专家计算完毕后，强制注入原始位置编码
        out_i = out_i + pe               # [B, T, d]
        expert_outputs.append(out_i)

    # 加权聚合
    expert_stack = torch.stack(expert_outputs, dim=1)  # [B, N, T, d]
    w = gate_weights[:, :, None, None]                  # [B, N, 1, 1]
    moe_out = (expert_stack * w).sum(dim=1)             # [B, T, d]
    return moe_out
```

---

## 模块4：残差连接与骨干适配

MoE 模块插入骨干 Transformer 的**第 `L` 层之后**（`L` 由 config 指定，默认为最后一层）：

```python
# h_L: 骨干第 L 层输出 [B, T, d]
# moe_out: MoE 聚合输出 [B, T, d]
h_prime = self.residual_ln(h_L + moe_out)   # LayerNorm(h_L + moe_out)

# h_prime 送入骨干第 L+1 层（若 L 为最后层则直接用于预测头）
```

---

## 模块5：损失函数

总损失为：

```
L_total = L_task + λ_balance * L_balance + λ_diversity * L_diversity + λ_entropy * L_entropy
```

各项默认权重：`λ_balance = 0.1`，`λ_diversity = 0.05`，`λ_entropy = 0.01`（均由 config 传入）。

### 5.1 `L_task`（任务主损失）

与原 `CPL_MoE` 保持一致，不做修改。

### 5.2 `L_balance`（Macro/Micro 组间负载均衡损失）

防止所有样本涌向同一组专家：

```python
# gate_weights [B, N]
macro_load = gate_weights[:, :N//2].sum(dim=-1).mean()   # 批内 Macro 组平均负载
micro_load  = gate_weights[:, N//2:].sum(dim=-1).mean()  # 批内 Micro 组平均负载

# 重要性加权：长文本样本加大约束力度
# query_len [B]: 有效 token 数（mask.sum(dim=-1)）
mean_len = query_len.float().mean()
w_i = 1.0 + beta * (query_len.float() > mean_len).float()   # [B]，beta=0.5 默认

# 加权后各样本的 Macro 偏置期望：长文本应多走 Macro
# target_macro_ratio: 超参数，默认 0.6（Macro 组期望承载 60% 负载）
macro_load_per_sample = gate_weights[:, :N//2].sum(dim=-1)   # [B]
L_balance = (w_i * (macro_load_per_sample - target_macro_ratio).pow(2)).mean()
```

**注意**：`w_i` 是对"偏离 Macro 目标"的惩罚加权，**不是**对路由正确性的监督（弱监督下不存在路由 ground-truth）。

### 5.3 `L_diversity`（组间多样性损失）

鼓励 Macro/Micro 组的路由分布在 batch 内存在显著差异：

```python
# 计算 batch 内 Macro 和 Micro 组的平均分布
p_macro = gate_weights[:, :N//2].mean(dim=0)    # [N//2]
p_micro  = gate_weights[:, N//2:].mean(dim=0)   # [N//2]

# 期望两组分布相异：最大化 JS 散度（等价于最小化负 JS 散度）
p_macro = p_macro / p_macro.sum()
p_micro  = p_micro  / p_micro.sum()
m = 0.5 * (p_macro + p_micro)
js_div = 0.5 * F.kl_div(m.log(), p_macro, reduction='sum') \
       + 0.5 * F.kl_div(m.log(), p_micro,  reduction='sum')
L_diversity = -js_div   # 最小化负 JS 散度，即最大化两组差异
```

**注意**：`L_balance` 和 `L_diversity` 目标不同（前者控制负载比例，后者控制分布差异），梯度方向独立，用独立权重系数 `λ_balance` 和 `λ_diversity` 调控，不会相互干扰。

### 5.4 `L_entropy`（路由熵正则）

防止路由过于 one-hot（失去软聚合优势）：

```python
# gate_weights [B, N]
H = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1)   # [B]
L_entropy = -H.mean()   # 最小化负熵，即鼓励适度分散
```

---

## 模块6：训练策略实现

### 6.1 两阶段训练控制

在 `CPL_MoEv2.__init__` 中添加：

```python
self.gating_frozen = True   # 初始冻结门控网络
```

在训练脚本或 `train_step` 中（需在 `cpl_moev2.py` 暴露接口）：

```python
def set_training_stage(self, stage: int):
    """
    stage=1: Warm-up，冻结门控网络，专家均等权重训练
    stage=2: Joint Training，解冻门控网络，启用所有损失项
    """
    if stage == 1:
        for p in self.gating_network.parameters():
            p.requires_grad_(False)
        self.gating_frozen = True
    elif stage == 2:
        for p in self.gating_network.parameters():
            p.requires_grad_(True)
        self.gating_frozen = False
```

当 `gating_frozen=True` 时，`gate_weights` 强制设为均匀分布 `1/N`，不经过 MLP 计算。

### 6.2 路由噪声退火

在 `GatingNetwork` 中维护噪声标准差：

```python
self.noise_std = nn.Parameter(torch.tensor(0.5), requires_grad=False)   # 初始 0.5

def anneal_noise(self, current_step: int, total_warmup_steps: int):
    """在训练脚本中每步调用，线性退火到 0"""
    ratio = min(current_step / total_warmup_steps, 1.0)
    self.noise_std.fill_(0.5 * (1.0 - ratio))
```

### 6.3 Scale Prior 权重预热

`alpha`（Scale Prior 强度）从 0 线性预热到目标值：

```python
# alpha 初始化为 0，在 set_training_stage(2) 后开始预热
def warmup_alpha(self, current_step: int, warmup_steps: int, target_alpha: float = 1.0):
    ratio = min(current_step / warmup_steps, 1.0)
    self.gating_network.alpha.fill_(target_alpha * ratio)
```

---

## 模块7：推理阶段行为

```python
def forward(self, video_feat, query_feat, query_mask, mode='train'):
    ...
    if mode == 'eval':
        # 推理使用确定性路由：取 Top-2 专家，其余权重置 0，归一化
        top2_vals, top2_idx = gate_weights.topk(2, dim=-1)
        gate_weights_hard = torch.zeros_like(gate_weights)
        gate_weights_hard.scatter_(1, top2_idx, top2_vals)
        gate_weights_hard = gate_weights_hard / gate_weights_hard.sum(dim=-1, keepdim=True)
        gate_weights = gate_weights_hard
    ...
```

---

## 接口约定（与原 cpl_moe.py 保持一致）

```python
class CPL_MoEv2(nn.Module):
    def __init__(self, config):
        """
        config 新增字段（其余与 CPL_MoE 相同）：
          - num_experts: int，专家总数 N（需为偶数，默认 4）
          - d_scale: int，尺度特征维度（默认 d_model // 4）
          - lambda_balance: float（默认 0.1）
          - lambda_diversity: float（默认 0.05）
          - lambda_entropy: float（默认 0.01）
          - beta: float，长文本重要性加权系数（默认 0.5）
          - target_macro_ratio: float（默认 0.6）
          - moe_insert_layer: int，MoE 插入骨干第几层后（默认 -1 即最后层）
          - noise_init_std: float（默认 0.5）
          - alpha_target: float（默认 1.0）
        """

    def forward(self, video_feat, query_feat, query_mask,
                video_pe=None, mode='train'):
        """
        输入:
          video_feat [B, T, d_model]
          query_feat [B, L, d_model]
          query_mask [B, L]         1=有效 token
          video_pe   [B, T, d_model] 原始时序位置编码（可选，None 时内部生成）
          mode       'train' | 'eval'

        输出 (train):
          pred_start [B, T], pred_end [B, T]  （与原 CPL_MoE 格式一致）
          loss_dict: {
              'loss_task': ...,
              'loss_balance': ...,
              'loss_diversity': ...,
              'loss_entropy': ...,
              'loss_total': ...,
              'scale_score': ...,          # 用于监控
              'macro_load': ...,           # 用于监控
              'routing_weights': ...       # [B, N]，用于可视化
          }

        输出 (eval):
          pred_start [B, T], pred_end [B, T]
        """
```

---

## 消融实验开关（供后续实验使用）

在 `config` 中添加布尔开关，允许逐模块消融：

```python
config.use_scale_estimator   = True   # 模块1
config.use_stratified_routing = True  # 模块2（专家结构归纳偏置）
config.use_scale_prior_bias  = True   # Scale Prior Bias
config.use_cross_attn_fusion = True   # 模块3跨模态 Cross-Attention
config.use_balance_loss      = True   # L_balance
config.use_diversity_loss    = True   # L_diversity
config.use_entropy_loss      = True   # L_entropy
```

当某个开关关闭时，对应模块退化为其平凡实现（如关闭 `use_scale_estimator` 时，`scale_score` 恒为 0.5）。

---

## 路由可视化接口（供调试使用）

```python
def get_routing_stats(self, video_feat, query_feat, query_mask):
    """
    返回:
      routing_weights [B, N]: 每个样本的专家分配权重
      scale_scores [B]: 每个样本的尺度得分
      macro_ratio [B]: 每个样本流向 Macro 组的总权重
    用于按 gt_coverage_bucket 分组统计路由行为。
    """
```
