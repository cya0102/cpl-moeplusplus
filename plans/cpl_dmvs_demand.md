# Role: 高级计算机视觉与深度学习专家

# Task: 将 DMVS (Decoupled Motion Expression Video Segmentation) 的多模态解耦思想融入到 CPL (Contrastive Proposal Learning) 的 PyTorch 实现中。
**重要限制**：为了不影响现有的工程，请你编写一个全新的完整文件 `cpl_dmvs.py`。请将新的模型类命名为 `CPL_DMVS`，而不是覆盖原有的 `cpl.py`。

# Background:
我们要基于原版 CPL 模型进行结构化创新。原版 CPL 是一个基于 Transformer 的视频文本 Grounding 模型，输入特征是预提取的 1D 时序特征（例如 I3D/C3D），形状为 `[B, T, C]`。
现需要引入以下两大创新点：
1. **基于 STAF 思想的纯视频特征聚合 (Video Query Generation)**：利用自关系建模（Self-relation Modeling）和动态元学习器（Meta-Learner），将帧级特征聚合为全局视频级查询 `Q_v`。
2. **DMVS 文本/视觉接耦交互 (Decoupled Cross-Modal Interaction)**：将文本解耦为静态词和运动词，让静态文本线索引导帧级查询 `Q_f`，运动文本线索引导视频级查询 `Q_v`，最后送入 CPL 的双流 Transformer。

---

# Specific Implementation Steps (代码编写要求):

请根据以下要求，为 `CPL_DMVS` 类提供完整的 `__init__` 和 `forward` 方法实现，并补全必要的依赖导入。请加上详细的中文注释和张量维度说明。

## 1. `__init__` 模块设计
在原版 CPL 初始化的基础上（保留 `frame_fc`, `word_fc`, `trans` 等），新增以下子模块（设 hidden_size = config['hidden_size']）：

- **动态元聚合模块 (STAF-based Aggregation)**:
  - 两层 MLP 作为 Meta-Learner：`Linear1 (T -> hidden_size) -> ReLU -> Linear2 (hidden_size -> T)`（注：若动态适配不同 T 有难度，可改用自适应池化将 T 统一后再输入 MLP，或者使用 1D 卷积）。
- **DMVS 解耦交互模块**:
  - `self.static_cross_attn`: `nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)`，用于帧级特征与静态文本交互。
  - `self.motion_cross_attn`: `nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)`，用于视频级特征与运动文本交互。

## 2. `forward` 方法签名
- 新增两个形参：`static_mask=None, motion_mask=None`。这两个掩码的形状与 `words_mask` 相同，布尔值 `True` 代表该词属于静态/运动词汇。

## 3. 视频特征处理：提取 Q_f 与 Q_v
在 `frames_feat = self.frame_fc(frames_feat)` 处理之后（此时形状为 `[B, T, C]`）：
- **帧级查询 $Q_f$**: 直接使用当前的 `frames_feat`。
- **视频级查询 $Q_v$ (基于 STAF 思想适配到 1D 时序)**:
  1. **Self-relation**: 计算帧之间的自关系矩阵 `M = torch.bmm(frames_feat, frames_feat.transpose(1, 2))`，形状 `[B, T, T]`。
  2. 行级池化得到 `M_bar` = `M.mean(dim=-1)`，形状 `[B, T]`。
  3. **Meta-Learner**: 将 `M_bar` 送入两层 MLP 输出动态权重 `d`，形状 `[B, T]`。
  4. **Attention Generation**: 将 `d` 扩展后与 `M` 逐点相乘：`attn_logits = d.unsqueeze(2) * M`。沿 `dim=-1` 做 Softmax，得到 `A_self`，形状 `[B, T, T]`。
  5. **Residual & Global Pooling**: 计算增强特征 `R_enhanced = frames_feat + torch.bmm(A_self, frames_feat)`。对 `R_enhanced` 沿时间维度 T 结合 `frames_mask` 进行全局平均池化，最终得到全局视频查询 $Q_v$，形状为 `[B, 1, C]`。

## 4. DMVS 解耦跨模态交互
- 处理文本掩码：结合原有的 `words_mask`（0代表padding）和传入的 `static_mask/motion_mask`，构造出适合 PyTorch MultiheadAttention 的 `key_padding_mask`（True 代表忽略该 Token）。
- **静态感知**: 使用 `self.static_cross_attn` 让 $Q_f$ `[B, T, C]` 作为 Query 关注 `words_feat`。加上残差得到 $Q_f'$。
- **运动感知**: 使用 `self.motion_cross_attn` 让 $Q_v$ `[B, 1, C]` 作为 Query 关注 `words_feat`。加上残差得到 $Q_v'$。

## 5. 重组序列
- 彻底废弃 CPL 原本拼接在末尾的零向量 `pred_vec`，替换为注入了全局动作感知的 $Q_v'$。
- 拼接序列：`frames_feat_new = torch.cat([Q_f_prime, Q_v_prime], dim=1)`，形状为 `[B, T+1, C]`。
- 同步在 `frames_mask` 末尾追加一个始终为有效 (True/1) 的位置掩码。
- 将新序列和新掩码送入 `self.trans(...)`。

# Output Style
请不要省略代码！请输出 `cpl_dmvs.py` 文件的**全部**代码（包含完整的 imports、辅助函数如 `_generate_mask` 和位置编码类），确保我可以一键复制并直接运行。