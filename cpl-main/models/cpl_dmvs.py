"""
CPL_DMVS: CPL with Decoupled Motion-Visual Segmentation

将 DMVS (Decoupled Motion Expression Video Segmentation) 的多模态解耦思想融入 CPL。
核心创新:
1. 基于 STAF 思想的 Self-relation + Meta-Learner 视频特征聚合，生成全局视频级查询 Q_v
2. DMVS 文本/视觉解耦交互: 静态词引导帧级查询 Q_f，运动词引导视频级查询 Q_v
3. 用 Q_v' 替代原版 CPL 的可学习 pred_vec，注入全局动作感知信息
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
import math


class CPL_DMVS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']
        hidden_size = config['hidden_size']
        self.hidden_size = hidden_size

        # ==================== 原版 CPL 基础组件 ====================
        # 帧特征投影: [B, T, frames_input_size] -> [B, T, hidden_size]
        self.frame_fc = nn.Linear(config['frames_input_size'], hidden_size)
        # 词特征投影: [B, L, words_input_size] -> [B, L, hidden_size]
        self.word_fc = nn.Linear(config['words_input_size'], hidden_size)
        # 可学习的特殊 token
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        # 注意: 不再使用 pred_vec，由 Q_v' 替代

        # 双流 Transformer
        self.trans = DualTransformer(**config['DualTransformer'])
        # 语义补全头
        self.fc_comp = nn.Linear(hidden_size, self.vocab_size)
        # 高斯参数预测头
        self.fc_gauss = nn.Linear(hidden_size, self.num_props * 2)
        # 词位置编码
        self.word_pos_encoder = SinusoidalPositionalEmbedding(hidden_size, 0, 20)

        # ==================== 新增: STAF-based 动态元聚合模块 ====================
        # Meta-Learner: 对每帧的自关系统计量 (标量) 生成动态权重
        # 输入: M_bar 的每个元素 [B, T, 1] -> 输出: 动态权重 [B, T, 1]
        # 使用逐元素 MLP，不依赖固定 T，适配变长视频
        self.meta_learner = nn.Sequential(
            nn.Linear(1, hidden_size),   # 标量 -> hidden_size
            nn.ReLU(),
            nn.Linear(hidden_size, 1)    # hidden_size -> 标量权重
        )

        # ==================== 新增: DMVS 解耦跨模态交互模块 ====================
        # 静态交叉注意力: Q_f (帧级查询) 关注静态词 (名词、形容词)
        self.static_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        # 运动交叉注意力: Q_v (视频级查询) 关注运动词 (动词、副词)
        self.motion_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights,
                static_mask=None, motion_mask=None, **kwargs):
        """
        Args:
            frames_feat: [B, T, frame_dim] 视频帧特征
            frames_len:  [B] 每个样本的有效帧数
            words_id:    [B, L] 词汇 ID
            words_feat:  [B, L+1, word_dim] 词嵌入 (第0位为 start token 占位)
            words_len:   [B] 每个样本的有效词数
            weights:     [B, L] 词权重 (用于 masked word 采样)
            static_mask: [B, L] 布尔掩码, True 表示该词是静态词 (名词/形容词)
            motion_mask: [B, L] 布尔掩码, True 表示该词是运动词 (动词/副词)
        """
        bsz, n_frames, _ = frames_feat.shape

        # ==================== 1. 帧特征处理 ====================
        # 注意: 与原版 CPL 不同，这里不拼接 pred_vec，后面会用 Q_v' 替代
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)  # [B, T, C]
        frames_mask = _generate_mask(frames_feat, frames_len)  # [B, T], 1=有效, 0=padding

        # ==================== 2. Q_f: 帧级查询 ====================
        Q_f = frames_feat  # [B, T, C]

        # ==================== 3. Q_v: 视频级查询 (STAF 聚合) ====================
        # 3.1 Self-relation: 计算帧间自关系矩阵
        # M[i,j] = <frames_feat[i], frames_feat[j]> 表示帧 i 和帧 j 的相似度
        M = torch.bmm(frames_feat, frames_feat.transpose(1, 2))  # [B, T, T]

        # 3.2 行级池化: 对每帧计算其与所有帧的平均相似度
        M_bar = M.mean(dim=-1)  # [B, T]

        # 3.3 Meta-Learner: 将标量 M_bar 送入逐元素 MLP 生成动态权重 d
        # M_bar.unsqueeze(-1): [B, T, 1] -> meta_learner -> [B, T, 1] -> squeeze -> [B, T]
        d = self.meta_learner(M_bar.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # 3.4 Attention Generation: d 与 M 逐点相乘后 Softmax
        # d.unsqueeze(2): [B, T, 1] 广播乘 M: [B, T, T] -> [B, T, T]
        attn_logits = d.unsqueeze(2) * M  # [B, T, T]

        # 对 padding 帧的列做 mask，防止 attention 泄漏到无效帧
        if frames_mask is not None:
            # frames_mask: [B, T], 0=padding -> 将 padding 列设为 -inf
            attn_mask = frames_mask.unsqueeze(1).float()  # [B, 1, T]
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))

        A_self = F.softmax(attn_logits, dim=-1)  # [B, T, T]
        # 处理全 -inf 行（全 padding 行 softmax 后为 nan）
        A_self = A_self.masked_fill(A_self != A_self, 0.0)

        # 3.5 Residual & Global Pooling
        # 增强特征: 原始帧特征 + 自关系聚合后的帧特征
        R_enhanced = frames_feat + torch.bmm(A_self, frames_feat)  # [B, T, C]

        # 结合 frames_mask 对 R_enhanced 做全局平均池化
        if frames_mask is not None:
            # frames_mask: [B, T] -> [B, T, 1]
            mask_expanded = frames_mask.unsqueeze(-1).float()  # [B, T, 1]
            R_masked = R_enhanced * mask_expanded  # 将 padding 帧特征置零
            Q_v = R_masked.sum(dim=1, keepdim=True) / (mask_expanded.sum(dim=1, keepdim=True) + 1e-6)
        else:
            Q_v = R_enhanced.mean(dim=1, keepdim=True)
        # Q_v: [B, 1, C] 全局视频级查询

        # ==================== 4. 文本特征处理 ====================
        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)  # [B, L+1, C]
        words_mask = _generate_mask(words_feat, words_len + 1)  # [B, L+1], 1=有效, 0=padding

        # ==================== 5. DMVS 解耦跨模态交互 ====================
        # 构造适合 nn.MultiheadAttention 的 key_padding_mask (True=忽略该位置)
        # words_feat 包含 start_token (位置0) + L 个词，words_mask 的有效范围是 [0, words_len]
        # static_mask / motion_mask 的维度是 [B, L]，对应 words_feat 的 [1:L+1] 位置

        # 对 words_feat 添加位置编码后用于 cross-attention
        words_feat_with_pos = words_feat + words_pos  # [B, L+1, C]

        if static_mask is not None:
            # 构造静态词的 key_padding_mask
            # start_token 位置: 设为忽略 (True)
            # 后续 L 个位置: 非静态词 or padding 设为忽略 (True)
            start_token_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=frames_feat.device)  # [B, 1]，不忽略 start 也可以，这里先忽略
            start_token_mask[:] = True  # 忽略 start token
            # static_mask: [B, L], True=静态词。需要截断到 words_feat 实际长度
            L_actual = words_feat_with_pos.size(1) - 1  # words_feat 长度减去 start token
            static_m = static_mask[:, :L_actual]  # [B, L_actual]
            # padding 位置也要忽略: words_mask[:, 1:] 中 0 表示 padding
            if words_mask is not None:
                valid_mask = words_mask[:, 1:L_actual+1].bool()  # [B, L_actual]
                # 最终: 该位置是有效的且是静态词才不忽略
                static_valid = static_m.bool() & valid_mask  # [B, L_actual]
            else:
                static_valid = static_m.bool()
            # nn.MultiheadAttention 的 key_padding_mask: True=忽略
            static_kpm = torch.cat([start_token_mask, ~static_valid], dim=1)  # [B, L+1]
            # 如果某个样本没有任何静态词，则不 mask（让所有词都可见，退化为全局 attention）
            all_masked = static_kpm.all(dim=1)  # [B]
            if all_masked.any():
                static_kpm[all_masked] = False
                if words_mask is not None:
                    static_kpm[all_masked] = ~words_mask[all_masked].bool()
        else:
            # 无 static_mask，使用全部词做 cross-attention
            if words_mask is not None:
                static_kpm = ~words_mask.bool()  # [B, L+1]
            else:
                static_kpm = None

        if motion_mask is not None:
            start_token_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=frames_feat.device)
            start_token_mask[:] = True
            L_actual = words_feat_with_pos.size(1) - 1
            motion_m = motion_mask[:, :L_actual]
            if words_mask is not None:
                valid_mask = words_mask[:, 1:L_actual+1].bool()
                motion_valid = motion_m.bool() & valid_mask
            else:
                motion_valid = motion_m.bool()
            motion_kpm = torch.cat([start_token_mask, ~motion_valid], dim=1)
            all_masked = motion_kpm.all(dim=1)
            if all_masked.any():
                motion_kpm[all_masked] = False
                if words_mask is not None:
                    motion_kpm[all_masked] = ~words_mask[all_masked].bool()
        else:
            if words_mask is not None:
                motion_kpm = ~words_mask.bool()
            else:
                motion_kpm = None

        # 5.1 静态感知: Q_f 关注静态词
        # Q_f: [B, T, C] (Query), words_feat_with_pos: [B, L+1, C] (Key, Value)
        Q_f_attn_out, _ = self.static_cross_attn(
            query=Q_f,
            key=words_feat_with_pos,
            value=words_feat_with_pos,
            key_padding_mask=static_kpm
        )
        Q_f_prime = Q_f + Q_f_attn_out  # [B, T, C] 残差连接

        # 5.2 运动感知: Q_v 关注运动词
        # Q_v: [B, 1, C] (Query), words_feat_with_pos: [B, L+1, C] (Key, Value)
        Q_v_attn_out, _ = self.motion_cross_attn(
            query=Q_v,
            key=words_feat_with_pos,
            value=words_feat_with_pos,
            key_padding_mask=motion_kpm
        )
        Q_v_prime = Q_v + Q_v_attn_out  # [B, 1, C] 残差连接

        # ==================== 6. 重组序列 ====================
        # 用 Q_v' 替代原版 CPL 的 pred_vec，拼接在帧序列末尾
        frames_feat = torch.cat([Q_f_prime, Q_v_prime], dim=1)  # [B, T+1, C]

        # 同步更新 frames_mask: 在末尾追加一个始终有效的位置
        if frames_mask is not None:
            ones = torch.ones(bsz, 1, dtype=frames_mask.dtype, device=frames_mask.device)
            frames_mask = torch.cat([frames_mask, ones], dim=1)  # [B, T+1]

        # ==================== 7. 生成高斯掩码 (与原版 CPL 一致) ====================
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        gauss_param = torch.sigmoid(self.fc_gauss(h[:, -1])).view(bsz * self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # ==================== 8. 下采样 + 提案特征 ====================
        props_len = n_frames // 4
        keep_idx = torch.linspace(0, n_frames - 1, steps=props_len).long()
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz * self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)

        # ==================== 9. 语义补全 (与原版 CPL 一致) ====================
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz * self.num_props, words_mask1.size(1), -1)

        pos_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1,
                                       decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)

        # ==================== 10. 负样本挖掘 (与原版 CPL 一致) ====================
        if self.use_negative:
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(
                props_len, gauss_center, gauss_width, kwargs['epoch']
            )

            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1,
                                    decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)

            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1,
                                    decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)

            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
        }

    def generate_gauss_weight(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))

        return weight / weight.max(dim=-1, keepdim=True)[0]

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma / 2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w / w1 * torch.exp(-(pos - c) ** 2 / (2 * w1 ** 2))
            return y1 / y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center - width / 2, min=0)
        left_center = left_width * min(epoch / self.max_epoch, 1) ** self.gamma * 0.5
        right_width = torch.clamp(1 - center - width / 2, min=0)
        right_center = 1 - right_width * min(epoch / self.max_epoch, 1) ** self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1 - right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1)
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1

        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码，支持任意长度序列。"""

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        return int(1e5)
