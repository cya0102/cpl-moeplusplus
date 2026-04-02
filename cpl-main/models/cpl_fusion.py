import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
import math


class CPL_Fusion(nn.Module):
    """
    CPL_Fusion: Fuses elements from CPL_DMVS and CPL_MoE into the base CPL architecture.

    Three gated augmentation modules (all initialized to zero = equivalent to base CPL at init):
      A. Gated Temporal Self-Attention on frames (from DMVS temporal modeling idea)
      B. POS Word-Type Embeddings (from DMVS word-type awareness, non-destructive)
      C. Query-Conditioned Proposal Refinement (from MoE query-guided idea, simplified)
    """

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

        # ===== Base CPL components (unchanged) =====
        self.frame_fc = nn.Linear(config['frames_input_size'], hidden_size)
        self.word_fc = nn.Linear(config['words_input_size'], hidden_size)
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(hidden_size, self.vocab_size)
        self.fc_gauss = nn.Linear(hidden_size, self.num_props * 2)

        self.word_pos_encoder = SinusoidalPositionalEmbedding(hidden_size, 0, 20)

        # ===== Fusion modules (gated, init to zero) =====
        fusion_config = config.get('fusion', {})

        # Module A: Gated Temporal Self-Attention
        self.use_temporal_attn = fusion_config.get('use_temporal_attn', False)
        if self.use_temporal_attn:
            temporal_heads = fusion_config.get('temporal_attn_heads', 4)
            self.temporal_self_attn = nn.MultiheadAttention(
                hidden_size, temporal_heads, batch_first=True, dropout=config['dropout']
            )
            self.temporal_norm = nn.LayerNorm(hidden_size)
            self.temporal_gate = nn.Parameter(torch.zeros(1))

        # Module B: POS Word-Type Embeddings
        self.use_word_type_embed = fusion_config.get('use_word_type_embed', False)
        if self.use_word_type_embed:
            self.word_type_embed = nn.Embedding(3, hidden_size)  # 0=other, 1=static, 2=motion
            self.type_embed_scale = fusion_config.get('type_embed_scale', 0.1)

        # Module C: Query-Conditioned Proposal Refinement
        self.use_proposal_refine = fusion_config.get('use_proposal_refine', False)
        if self.use_proposal_refine:
            self.fc_refine = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_props * 2)
            )
            self.refine_gate = nn.Parameter(torch.zeros(1))
            # Initialize last layer near zero for stable start
            nn.init.zeros_(self.fc_refine[-1].weight)
            nn.init.zeros_(self.fc_refine[-1].bias)

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights,
                static_mask=None, motion_mask=None, **kwargs):
        bsz, n_frames, _ = frames_feat.shape

        # ===== Step 1: Append pred_vec and project (same as base CPL) =====
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)  # [B, T+1, D_f]
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)  # [B, T+1, C]
        frames_mask = _generate_mask(frames_feat, frames_len)

        # ===== Step 2: [NEW] Gated temporal self-attention on frame positions only =====
        if self.use_temporal_attn:
            frame_part = frames_feat[:, :n_frames, :]  # [B, T, C]
            # Build key_padding_mask: True = ignore position
            if frames_mask is not None:
                temporal_kpm = ~frames_mask[:, :n_frames].bool()
            else:
                temporal_kpm = None
            temporal_out, _ = self.temporal_self_attn(
                frame_part, frame_part, frame_part,
                key_padding_mask=temporal_kpm
            )
            # Gated residual with LayerNorm
            gate_val = torch.sigmoid(self.temporal_gate)
            frame_part = frame_part + gate_val * self.temporal_norm(temporal_out)
            # Reassemble with pred_vec position untouched
            frames_feat = torch.cat([frame_part, frames_feat[:, n_frames:, :]], dim=1)

        # ===== Step 3: Process word features (same as base CPL) =====
        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)  # [B, L+1, C]
        words_mask = _generate_mask(words_feat, words_len + 1)

        # ===== Step 4: [NEW] Add POS word-type embeddings for decoder1 input =====
        if self.use_word_type_embed and static_mask is not None and motion_mask is not None:
            word_types = self._compute_word_types(static_mask, motion_mask, words_feat.size(1))
            type_embed = self.word_type_embed(word_types)  # [B, L+1, C]
            words_feat_d1 = words_feat + self.type_embed_scale * type_embed
        else:
            words_feat_d1 = words_feat

        # ===== Step 5: DualTransformer decoder1 (same as base CPL) =====
        enc_out, h = self.trans(
            frames_feat, frames_mask,
            words_feat_d1 + words_pos, words_mask,
            decoding=1
        )

        # ===== Step 6: Gaussian parameter prediction =====
        pred_feat = h[:, -1]  # [B, C] — the pred_vec position output
        gauss_logits = self.fc_gauss(pred_feat)  # [B, 2*num_props]

        # [NEW] Query-conditioned refinement
        if self.use_proposal_refine:
            query_feat = self._aggregate_query(words_feat, words_mask)  # [B, C]
            fused = torch.cat([pred_feat, query_feat], dim=-1)  # [B, 2C]
            delta = self.fc_refine(fused)  # [B, 2*num_props]
            gauss_logits = gauss_logits + torch.sigmoid(self.refine_gate) * delta

        gauss_param = torch.sigmoid(gauss_logits).view(bsz * self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # ===== Step 7: Downsample for efficiency (same as base CPL) =====
        props_len = n_frames // 4
        keep_idx = torch.linspace(0, n_frames - 1, steps=props_len).long()
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz * self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)

        # ===== Step 8: Semantic completion - mask words (same as base CPL) =====
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

        # ===== Step 9: DualTransformer decoder2 + fc_comp (same as base CPL) =====
        pos_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(
            props_feat, props_mask, words_feat1, words_mask1,
            decoding=2, gauss_weight=pos_weight, need_weight=True
        )
        words_logit = self.fc_comp(h)

        # ===== Step 10: Negative proposal mining (same as base CPL) =====
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

    def _compute_word_types(self, static_mask, motion_mask, seq_len):
        """
        Convert static_mask [B, L] and motion_mask [B, L] to word_types [B, seq_len].
        seq_len = L + 1 (start token at position 0).
        Types: 0=other/start, 1=static (noun/adj), 2=motion (verb/adv)
        """
        bsz = static_mask.size(0)
        word_types = torch.zeros(bsz, seq_len, dtype=torch.long, device=static_mask.device)
        L = static_mask.size(1)
        actual_L = min(L, seq_len - 1)
        # Position 0 is start token (type=0), positions 1..L are actual words
        static_part = static_mask[:, :actual_L].long()
        motion_part = motion_mask[:, :actual_L].long()
        # motion (type=2) takes precedence if both are set (shouldn't happen normally)
        word_types[:, 1:actual_L + 1] = static_part * 1 + motion_part * 2
        # Clamp to valid range [0, 2]
        word_types = word_types.clamp(0, 2)
        return word_types

    def _aggregate_query(self, words_feat, words_mask):
        """Masked mean pooling of word features for query representation."""
        if words_mask is not None:
            mask_expanded = words_mask.unsqueeze(-1).float()  # [B, L+1, 1]
            pooled = (words_feat * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            pooled = words_feat.mean(dim=1)
        return pooled  # [B, C]

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
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx,
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
                max_pos, self.embedding_dim, self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        return int(1e5)
