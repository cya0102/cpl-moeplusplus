"""
CPL_MoEv2: CPL with Scale-Aware Mixture of Experts

Key improvements over CPL_MoE:
1. Scale-Aware Gating System replacing naive gating network
2. Stratified Expert Pool (Macro + Micro experts) with structural inductive bias
3. Multi-scale temporal density analysis for video features
4. Scale prior bias for routing based on query/video scale signals
5. Auxiliary losses: balance, diversity, entropy regularization
6. Two-stage training with noise annealing and alpha warmup

Reference: CPL_MoEv2_final_spec.md
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.transformer import DualTransformer


# ==============================================================================
# Module 1: ScaleEstimator
# ==============================================================================

class QueryScaleEncoder(nn.Module):
    """
    Module 1.1: Extract query temporal scale features.
    
    Uses attention entropy and a learned temporal-keyword detector to estimate
    whether the query describes a long-range or short-range action.
    """

    def __init__(self, d_model, d_scale, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Lightweight self-attention for computing attention entropy
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )

        # Learned temporal keyword detector
        # (replaces hard-coded word list since we only have GloVe IDs, not strings)
        self.kw_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

        # Scale signal projection: [attn_entropy, kw_flag] -> d_scale
        self.scale_signal_proj = nn.Linear(2, d_scale)

    def forward(self, query_feat, query_mask):
        """
        Args:
            query_feat: [B, L, d_model]  projected word features (after word_fc)
            query_mask: [B, L]           1 = valid token, 0 = padding

        Returns:
            query_scale_feat: [B, d_scale]
            q_global:         [B, d_model]  mean-pooled query representation
        """
        # --- 1. Attention entropy ---
        key_padding_mask = ~query_mask.bool()  # True = padding for nn.MHA
        _, attn_weights = self.self_attn(
            query_feat, query_feat, query_feat,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights [B, nhead, L, L]
        )
        # Use first token's attention as the "CLS" proxy
        cls_attn = attn_weights[:, :, 0, :]                    # [B, nhead, L]
        entropy_per_head = -(cls_attn * (cls_attn + 1e-8).log()).sum(dim=-1)  # [B, nhead]
        attn_entropy = entropy_per_head.mean(dim=-1)            # [B]

        # --- 2. Temporal keyword score (learned) ---
        kw_scores = torch.sigmoid(
            self.kw_detector(query_feat).squeeze(-1)            # [B, L]
        )
        kw_scores = kw_scores * query_mask.float()
        kw_flag = kw_scores.max(dim=-1).values                  # [B]

        # --- 3. Fuse into scale feature ---
        scale_signal = torch.stack([attn_entropy, kw_flag], dim=-1)  # [B, 2]
        query_scale_feat = self.scale_signal_proj(scale_signal)      # [B, d_scale]

        # --- 4. Global query feature ---
        mask_f = query_mask.unsqueeze(-1).float()                    # [B, L, 1]
        q_global = (query_feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)

        return query_scale_feat, q_global


class VideoDensityAnalyzer(nn.Module):
    """
    Module 1.2: Multi-scale temporal density analysis for video features.
    
    Computes temporal difference energy at multiple scales and multi-resolution
    pooling features.
    """

    def __init__(self, d_model, d_scale):
        super().__init__()
        # 3 scales x 2 stats (mean, max) = 6
        self.density_proj = nn.Linear(6, d_scale)

    def forward(self, video_feat):
        """
        Args:
            video_feat: [B, T, d_model]

        Returns:
            video_density_feat: [B, d_scale]
            v_global:           [B, d_model]
            v_local:            [B, d_model]
        """
        B, T, d = video_feat.shape

        # --- 1. Multi-scale temporal difference energy ---
        stats = []
        for k in [1, 4, 16]:
            if k >= T:
                stats.extend([video_feat.new_zeros(B), video_feat.new_zeros(B)])
                continue
            diff_k = video_feat[:, k:, :] - video_feat[:, :-k, :]   # [B, T-k, d]
            energy_k = torch.norm(diff_k, dim=-1)                   # [B, T-k]
            stats.append(energy_k.mean(dim=-1))                      # [B]
            stats.append(energy_k.max(dim=-1).values)                # [B]

        density_raw = torch.stack(stats, dim=-1)                     # [B, 6]
        video_density_feat = self.density_proj(density_raw)          # [B, d_scale]

        # --- 2. Global video feature ---
        v_global = video_feat.mean(dim=1)                            # [B, d]

        # --- 3. Multi-resolution local features ---
        video_t = video_feat.transpose(1, 2)                         # [B, d, T]
        v_locals = []
        for w_div in [8, 4]:
            w = max(T // w_div, 1)
            stride = max(w // 2, 1)
            pooled = F.avg_pool1d(video_t, kernel_size=w, stride=stride)
            v_locals.append(pooled.mean(dim=-1))                     # [B, d]
        v_local = sum(v_locals) / len(v_locals)                      # [B, d]

        return video_density_feat, v_global, v_local


class ScaleEstimator(nn.Module):
    """
    Module 1: Combines QueryScaleEncoder + VideoDensityAnalyzer + ScaleScore.
    
    Produces a scalar scale_score in [0, 1] indicating whether the sample
    leans towards long-range (→1) or short-range (→0) temporal action.
    """

    def __init__(self, d_model, d_scale, nhead=4):
        super().__init__()
        self.query_encoder = QueryScaleEncoder(d_model, d_scale, nhead)
        self.video_analyzer = VideoDensityAnalyzer(d_model, d_scale)
        self.scale_proj = nn.Linear(2 * d_scale, 1)

    def forward(self, video_feat, query_feat, query_mask):
        """
        Returns:
            scale_score, query_scale_feat, video_density_feat,
            v_global, v_local, q_global
        """
        query_scale_feat, q_global = self.query_encoder(query_feat, query_mask)
        video_density_feat, v_global, v_local = self.video_analyzer(video_feat)

        fused = torch.cat([query_scale_feat, video_density_feat], dim=-1)
        scale_score = torch.sigmoid(self.scale_proj(fused)).squeeze(-1)  # [B]

        return scale_score, query_scale_feat, video_density_feat, v_global, v_local, q_global


# ==============================================================================
# Module 2: ScaleAwareGatingNetwork
# ==============================================================================

class LightCrossAttention(nn.Module):
    """Lightweight cross-attention (1-2 heads) for cross-modal conditioning."""

    def __init__(self, d_model, nhead=2, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query:            [B, 1, d]
            key / value:      [B, L, d]
            key_padding_mask: [B, L]  True = padding
        Returns:
            out: [B, 1, d]
        """
        out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        return out


class ScaleAwareGatingNetwork(nn.Module):
    """
    Module 2: Computes gate_weights [B, N] from scale-aware features.
    
    Includes cross-modal conditioning, feature compression, gate MLP,
    scale prior bias, and training noise.
    """

    def __init__(self, d_model, d_scale, num_experts,
                 use_cross_attn=True, use_scale_prior=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_scale = d_scale
        self.num_experts = num_experts
        self.use_cross_attn = use_cross_attn
        self.use_scale_prior = use_scale_prior

        # Cross-modal conditioning
        if use_cross_attn:
            self.cross_attn = LightCrossAttention(d_model, nhead=2, dropout=dropout)

        # Per-stream LayerNorm
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_qs = nn.LayerNorm(d_scale)
        self.ln_vd = nn.LayerNorm(d_scale)

        # Compress concatenated features → d_model
        gate_input_dim = 2 * d_model + 2 * d_scale
        self.compress = nn.Linear(gate_input_dim, d_model)

        # Gate MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
        )

        # Scale prior bias strength (warmed up from 0 → target)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Training noise std (annealed from init_std → 0)
        self.noise_std = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, v_global, q_global, query_feat, query_mask,
                query_scale_feat, video_density_feat, scale_score, training):
        """
        Args:
            v_global:           [B, d]
            q_global:           [B, d]
            query_feat:         [B, L, d]
            query_mask:         [B, L]  1 = valid
            query_scale_feat:   [B, d_scale]
            video_density_feat: [B, d_scale]
            scale_score:        [B]
            training:           bool

        Returns:
            gate_weights: [B, N]
        """
        B = v_global.size(0)
        N = self.num_experts
        device = v_global.device

        # Cross-modal conditioning
        if self.use_cross_attn:
            kpm = ~query_mask.bool()
            v_conditioned = self.cross_attn(
                v_global.unsqueeze(1), query_feat, query_feat,
                key_padding_mask=kpm,
            ).squeeze(1)                                      # [B, d]
        else:
            v_conditioned = v_global

        # LayerNorm each stream and concatenate
        g1 = self.ln_v(v_conditioned)                          # [B, d]
        g2 = self.ln_q(q_global)                               # [B, d]
        g3 = self.ln_qs(query_scale_feat)                      # [B, d_scale]
        g4 = self.ln_vd(video_density_feat)                    # [B, d_scale]

        gate_input = torch.cat([g1, g2, g3, g4], dim=-1)      # [B, 2d+2*d_scale]
        gate_input = F.relu(self.compress(gate_input))         # [B, d]

        # Gate MLP → raw logits
        logits = self.gate_mlp(gate_input)                     # [B, N]

        # Scale prior bias
        if self.use_scale_prior:
            half_n = N // 2
            bias = torch.zeros(B, N, device=device)
            bias[:, :half_n] = +self.alpha * scale_score.unsqueeze(-1)
            bias[:, half_n:] = -self.alpha * scale_score.unsqueeze(-1)
            logits = logits + bias

        # Training noise (annealed)
        if training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        gate_weights = F.softmax(logits, dim=-1)               # [B, N]
        return gate_weights

    # --- Training schedule helpers ---

    def anneal_noise(self, current_step, total_anneal_steps):
        """Linearly anneal noise_std from initial value to 0."""
        ratio = min(current_step / max(total_anneal_steps, 1), 1.0)
        init_std = 0.5  # matches __init__
        self.noise_std.fill_(init_std * (1.0 - ratio))

    def warmup_alpha(self, current_step, warmup_steps, target_alpha=1.0):
        """Linearly warm up alpha from 0 to target_alpha."""
        ratio = min(current_step / max(warmup_steps, 1), 1.0)
        self.alpha.fill_(target_alpha * ratio)


# ==============================================================================
# Module 3: StratifiedExpertPool
# ==============================================================================

class MacroExpert(nn.Module):
    """
    Macro expert: dilated self-attention for long-range temporal modelling.
    
    K/V are sub-sampled at stride = dilation, giving the expert a wider
    receptive field without increasing computation.
    """

    def __init__(self, d_model, dilation=2, nhead=4, dropout=0.1):
        super().__init__()
        self.dilation = dilation
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pe):
        """
        Args:
            x:  [B, T, d]
            pe: [B, T, d]  (not consumed directly; caller injects pe after)
        """
        B, T, d = x.shape

        # Dilated K/V
        k_idx = torch.arange(0, T, self.dilation, device=x.device)
        kv = x[:, k_idx]                                       # [B, T', d]

        # Self-attention with dilated K/V
        residual = x
        out, _ = self.attn(x, kv, kv)
        out = self.drop1(out)
        out = self.norm1(residual + out)

        # FFN
        residual = out
        out = self.ffn(out)
        out = self.drop2(out)
        out = self.norm2(residual + out)

        return out


class MicroExpert(nn.Module):
    """
    Micro expert: sliding-window self-attention for local temporal modelling.
    
    A 1-D convolution is applied to the Value projection to reinforce the
    local receptive field.  Window size is dynamic: T // 4.
    """

    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def _build_window_mask(T, window_size, device):
        """Return float additive mask [T, T]:  0 = attend, -inf = masked."""
        rows = torch.arange(T, device=device).unsqueeze(1)
        cols = torch.arange(T, device=device).unsqueeze(0)
        blocked = (rows - cols).abs() > (window_size // 2)
        mask = torch.zeros(T, T, device=device)
        mask.masked_fill_(blocked, float('-inf'))
        return mask

    def forward(self, x, pe):
        """
        Args:
            x:  [B, T, d]
            pe: [B, T, d]
        """
        B, T, d = x.shape
        window_size = max(T // 4, 1)

        attn_mask = self._build_window_mask(T, window_size, x.device)  # [T, T]

        # Conv-based value projection
        v = self.conv_v(x.transpose(1, 2)).transpose(1, 2)             # [B, T, d]

        # Windowed self-attention
        residual = x
        out, _ = self.attn(x, x, v, attn_mask=attn_mask)
        out = self.drop1(out)
        out = self.norm1(residual + out)

        # FFN
        residual = out
        out = self.ffn(out)
        out = self.drop2(out)
        out = self.norm2(residual + out)

        return out


class StratifiedExpertPool(nn.Module):
    """
    Module 3: Pool of Macro (first half) and Micro (second half) experts.
    
    Positional encoding is re-injected after each expert to prevent
    temporal information loss.
    """

    def __init__(self, d_model, num_experts, nhead=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        half_n = num_experts // 2

        self.experts = nn.ModuleList()

        # Macro experts (dilated attention, alternating dilation ∈ {2, 4})
        dilations = [2, 4]
        for i in range(half_n):
            d = dilations[i % len(dilations)]
            self.experts.append(MacroExpert(d_model, dilation=d, nhead=nhead, dropout=dropout))

        # Micro experts (sliding-window attention)
        for _ in range(num_experts - half_n):
            self.experts.append(MicroExpert(d_model, nhead=nhead, dropout=dropout))

    def forward(self, video_feat, gate_weights, pe):
        """
        Args:
            video_feat:   [B, T, d]
            gate_weights: [B, N]
            pe:           [B, T, d]  positional encoding to re-inject

        Returns:
            moe_out: [B, T, d]
        """
        expert_outputs = []
        for expert in self.experts:
            out_i = expert(video_feat, pe)
            out_i = out_i + pe  # positional encoding protection
            expert_outputs.append(out_i)

        expert_stack = torch.stack(expert_outputs, dim=1)       # [B, N, T, d]
        w = gate_weights[:, :, None, None]                      # [B, N, 1, 1]
        moe_out = (expert_stack * w).sum(dim=1)                 # [B, T, d]
        return moe_out


# ==============================================================================
# Main Model
# ==============================================================================

class CPL_MoEv2(nn.Module):
    """
    CPL with Scale-Aware Mixture of Experts (v2).

    Replaces the naive gating in CPL_MoE with a Scale-Aware Gating System
    (ScaleEstimator + ScaleAwareGatingNetwork + StratifiedExpertPool) that
    enhances video temporal features before proposal prediction.
    """

    def __init__(self, config):
        super().__init__()

        # ---- Standard CPL hyper-parameters ----
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config['sigma']
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']
        self.hidden_size = config['hidden_size']

        # ---- MoEv2 hyper-parameters ----
        mc = config.get('MoEv2', {})
        self.num_experts = mc.get('num_experts', 4)
        assert self.num_experts % 2 == 0, "num_experts must be even (Macro/Micro split)"
        self.d_scale = mc.get('d_scale', self.hidden_size // 4)
        self.lambda_balance = mc.get('lambda_balance', 0.1)
        self.lambda_diversity = mc.get('lambda_diversity', 0.05)
        self.lambda_entropy = mc.get('lambda_entropy', 0.01)
        self.beta = mc.get('beta', 0.5)
        self.target_macro_ratio = mc.get('target_macro_ratio', 0.6)
        self.alpha_target = mc.get('alpha_target', 1.0)
        self.noise_init_std = mc.get('noise_init_std', 0.5)

        # Ablation switches
        self.use_scale_estimator = mc.get('use_scale_estimator', True)
        self.use_stratified_routing = mc.get('use_stratified_routing', True)
        self.use_scale_prior_bias = mc.get('use_scale_prior_bias', True)
        self.use_cross_attn_fusion = mc.get('use_cross_attn_fusion', True)
        self.use_balance_loss = mc.get('use_balance_loss', True)
        self.use_diversity_loss = mc.get('use_diversity_loss', True)
        self.use_entropy_loss = mc.get('use_entropy_loss', True)

        # ---- Feature projection layers ----
        self.frame_fc = nn.Linear(config['frames_input_size'], self.hidden_size)
        self.word_fc = nn.Linear(config['words_input_size'], self.hidden_size)

        # Learnable special tokens
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        # ---- Dual Transformer backbone ----
        self.trans = DualTransformer(**config['DualTransformer'])

        # ---- Prediction heads ----
        self.fc_comp = nn.Linear(self.hidden_size, self.vocab_size)
        self.fc_gauss = nn.Linear(self.hidden_size, self.num_props * 2)

        # ---- Positional embeddings ----
        self.word_pos_encoder = SinusoidalPositionalEmbedding(self.hidden_size, 0, 20)
        self.video_pos_encoder = SinusoidalPositionalEmbedding(self.hidden_size, 0, 1024)

        # ---- MoEv2 modules ----
        self.scale_estimator = ScaleEstimator(
            self.hidden_size, self.d_scale, nhead=4
        )
        self.gating_network = ScaleAwareGatingNetwork(
            self.hidden_size, self.d_scale, self.num_experts,
            use_cross_attn=self.use_cross_attn_fusion,
            use_scale_prior=self.use_scale_prior_bias,
            dropout=config['dropout'],
        )
        self.expert_pool = StratifiedExpertPool(
            self.hidden_size, self.num_experts, nhead=4, dropout=config['dropout'],
        )

        # Residual LayerNorm (Module 4)
        self.residual_ln = nn.LayerNorm(self.hidden_size)

        # ---- Two-stage training control ----
        self.gating_frozen = True  # Stage 1 by default

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, frames_feat, frames_len, words_id, words_feat,
                words_len, weights, **kwargs):
        """
        Matches the call signature of CPL / CPL_MoE so that the existing
        runner can use this model as a drop-in replacement.
        """
        bsz, n_frames, _ = frames_feat.shape

        # ---- 1. Feature projection (identical to CPL) ----
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)                 # [B, T+1, d]
        frames_mask = _generate_mask(frames_feat, frames_len)    # [B, T+1]

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)                    # [B, L, d]
        words_mask = _generate_mask(words_feat, words_len + 1)  # [B, L]

        # Save projected (unmasked) words for scale estimator
        words_feat_proj = words_feat                              # [B, L, d]
        words_mask_full = words_mask                              # [B, L]

        # ---- 2. Cross-modal encoding via DualTransformer ----
        enc_out, h = self.trans(
            frames_feat, frames_mask,
            words_feat + words_pos, words_mask,
            decoding=1,
        )
        # h: [B, T+1, d]  (frame positions + pred_vec position)

        # ---- 3. Separate video features ----
        video_h = h[:, :n_frames]                                # [B, T, d]
        pred_h = h[:, -1]                                        # [B, d]

        # ---- 4. Video positional encoding ----
        video_pe = self.video_pos_encoder(video_h)               # [1, T, d]
        video_pe = video_pe.expand(bsz, -1, -1)                 # [B, T, d]

        # ==================================================================
        # Scale-Aware MoE System
        # ==================================================================

        # -- Module 1: ScaleEstimator --
        if self.use_scale_estimator:
            (scale_score, query_scale_feat, video_density_feat,
             v_global, v_local, q_global) = self.scale_estimator(
                video_h, words_feat_proj, words_mask_full
            )
        else:
            # Ablation: constant scale
            scale_score = torch.full((bsz,), 0.5, device=video_h.device)
            mask_f = words_mask_full.unsqueeze(-1).float()
            q_global = (words_feat_proj * mask_f).sum(1) / (mask_f.sum(1) + 1e-6)
            v_global = video_h.mean(dim=1)
            query_scale_feat = torch.zeros(bsz, self.d_scale, device=video_h.device)
            video_density_feat = torch.zeros(bsz, self.d_scale, device=video_h.device)

        # -- Module 2: Scale-Aware Gating --
        if self.gating_frozen:
            gate_weights = torch.ones(
                bsz, self.num_experts, device=video_h.device
            ) / self.num_experts
        else:
            gate_weights = self.gating_network(
                v_global, q_global,
                words_feat_proj, words_mask_full,
                query_scale_feat, video_density_feat,
                scale_score, self.training,
            )

        # Inference: deterministic Top-2 routing
        if not self.training:
            top2_vals, top2_idx = gate_weights.topk(2, dim=-1)
            gw_hard = torch.zeros_like(gate_weights)
            gw_hard.scatter_(1, top2_idx, top2_vals)
            gate_weights = gw_hard / (gw_hard.sum(dim=-1, keepdim=True) + 1e-8)

        # -- Module 3: Stratified Expert Pool --
        moe_out = self.expert_pool(video_h, gate_weights, video_pe)   # [B, T, d]

        # -- Module 4: Residual + LayerNorm --
        enhanced_video = self.residual_ln(video_h + moe_out)          # [B, T, d]

        # ==================================================================
        # Gaussian proposal prediction
        # ==================================================================
        # Pool enhanced video features and combine with prediction token
        video_mask_f = frames_mask[:, :n_frames].unsqueeze(-1).float()  # [B, T, 1]
        moe_pooled = (moe_out * video_mask_f).sum(1) / (video_mask_f.sum(1) + 1e-6)
        multimodal_feat = pred_h + moe_pooled                          # [B, d]

        gauss_param = torch.sigmoid(
            self.fc_gauss(multimodal_feat)
        ).view(bsz * self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # ==================================================================
        # Downsample for efficiency
        # ==================================================================
        props_len = n_frames // 4
        keep_idx = torch.linspace(0, n_frames - 1, steps=props_len).long()
        enhanced_frames = enhanced_video[:, keep_idx]                  # [B, T', d]
        frames_mask_ds = frames_mask[:, keep_idx]                      # [B, T']

        props_feat = (enhanced_frames.unsqueeze(1)
                      .expand(bsz, self.num_props, -1, -1)
                      .contiguous()
                      .view(bsz * self.num_props, props_len, -1))
        props_mask = (frames_mask_ds.unsqueeze(1)
                      .expand(bsz, self.num_props, -1)
                      .contiguous()
                      .view(bsz * self.num_props, -1))

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)

        # ==================================================================
        # Semantic completion
        # ==================================================================
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = (words_mask.unsqueeze(1)
                       .expand(bsz, self.num_props, -1)
                       .contiguous()
                       .view(bsz * self.num_props, -1))
        words_id1 = (words_id.unsqueeze(1)
                     .expand(bsz, self.num_props, -1)
                     .contiguous()
                     .view(bsz * self.num_props, -1))
        words_feat1 = (words_feat.unsqueeze(1)
                       .expand(bsz, self.num_props, -1, -1)
                       .contiguous()
                       .view(bsz * self.num_props, words_mask1.size(1), -1))

        pos_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h_comp, attn_weight = self.trans(
            props_feat, props_mask, words_feat1, words_mask1,
            decoding=2, gauss_weight=pos_weight, need_weight=True,
        )
        words_logit = self.fc_comp(h_comp)

        # ==================================================================
        # Negative proposal mining
        # ==================================================================
        if self.use_negative:
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(
                props_len, gauss_center, gauss_width, kwargs['epoch']
            )

            _, neg_h_1 = self.trans(
                props_feat, props_mask, words_feat1, words_mask1,
                decoding=2, gauss_weight=neg_1_weight,
            )
            neg_words_logit_1 = self.fc_comp(neg_h_1)

            _, neg_h_2 = self.trans(
                props_feat, props_mask, words_feat1, words_mask1,
                decoding=2, gauss_weight=neg_2_weight,
            )
            neg_words_logit_2 = self.fc_comp(neg_h_2)

            _, ref_h = self.trans(
                enhanced_frames, frames_mask_ds, words_feat, words_mask,
                decoding=2,
            )
            ref_words_logit = self.fc_comp(ref_h)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        # ==================================================================
        # MoEv2 auxiliary losses
        # ==================================================================
        aux_loss = None
        if self.training and not self.gating_frozen:
            aux_loss = torch.tensor(0.0, device=video_h.device, requires_grad=True)

            if self.use_balance_loss:
                l_bal = self._compute_balance_loss(gate_weights, words_len)
                aux_loss = aux_loss + self.lambda_balance * l_bal

            if self.use_diversity_loss:
                l_div = self._compute_diversity_loss(gate_weights)
                aux_loss = aux_loss + self.lambda_diversity * l_div

            if self.use_entropy_loss:
                l_ent = self._compute_entropy_loss(gate_weights)
                aux_loss = aux_loss + self.lambda_entropy * l_ent

        # ==================================================================
        # Return
        # ==================================================================
        return {
            # Required by rec_loss / ivc_loss
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            # MoEv2 auxiliary loss
            'aux_loss': aux_loss,
            # Monitoring (detached, not in computation graph)
            'scale_score': scale_score.detach(),
            'gate_weights_monitor': gate_weights.detach(),
            'macro_load': gate_weights[:, :self.num_experts // 2].sum(dim=-1).mean().item(),
        }

    # ------------------------------------------------------------------
    # MoEv2 Loss Functions
    # ------------------------------------------------------------------

    def _compute_balance_loss(self, gate_weights, words_len):
        """
        L_balance: encourage target Macro/Micro load ratio, with
        importance weighting for long queries.
        """
        half_n = self.num_experts // 2
        macro_load = gate_weights[:, :half_n].sum(dim=-1)        # [B]

        # Importance weighting: penalise long-query samples more
        query_len = words_len.float()
        mean_len = query_len.mean()
        w_i = 1.0 + self.beta * (query_len > mean_len).float()  # [B]

        loss = (w_i * (macro_load - self.target_macro_ratio).pow(2)).mean()
        return loss

    def _compute_diversity_loss(self, gate_weights):
        """
        L_diversity: maximise JS-divergence between Macro and Micro
        group distributions (averaged over batch).
        """
        half_n = self.num_experts // 2
        p_macro = gate_weights[:, :half_n].mean(dim=0)          # [half_n]
        p_micro = gate_weights[:, half_n:].mean(dim=0)          # [half_n]

        # Normalise to valid distributions
        p_macro = p_macro / (p_macro.sum() + 1e-8)
        p_micro = p_micro / (p_micro.sum() + 1e-8)

        m = 0.5 * (p_macro + p_micro)
        js_div = (0.5 * F.kl_div(m.log(), p_macro, reduction='sum')
                  + 0.5 * F.kl_div(m.log(), p_micro, reduction='sum'))

        # Minimise negative JS  ⟹  maximise diversity
        return -js_div

    def _compute_entropy_loss(self, gate_weights):
        """
        L_entropy: prevent routing from collapsing to one-hot by
        encouraging moderate dispersion.
        """
        H = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1)  # [B]
        return -H.mean()  # minimise negative entropy → encourage spread

    # ------------------------------------------------------------------
    # Two-stage training interface
    # ------------------------------------------------------------------

    def set_training_stage(self, stage):
        """
        stage 1: Warm-up — freeze gating, experts train with uniform weights.
        stage 2: Joint   — unfreeze gating, enable all losses.
        """
        if stage == 1:
            for p in self.gating_network.parameters():
                p.requires_grad_(False)
            self.gating_frozen = True
        elif stage == 2:
            for p in self.gating_network.parameters():
                p.requires_grad_(True)
            self.gating_frozen = False
        else:
            raise ValueError(f"Unknown stage {stage}")

    def anneal_noise(self, current_step, total_anneal_steps):
        self.gating_network.anneal_noise(current_step, total_anneal_steps)

    def warmup_alpha(self, current_step, warmup_steps):
        self.gating_network.warmup_alpha(
            current_step, warmup_steps, target_alpha=self.alpha_target
        )

    # ------------------------------------------------------------------
    # Routing visualisation / debug
    # ------------------------------------------------------------------

    def get_routing_stats(self, frames_feat, frames_len, words_feat, words_len):
        """
        Utility for offline analysis — returns routing weights and scale
        scores without running the full forward (no loss / completion).
        """
        bsz, n_frames, _ = frames_feat.shape

        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        _, h = self.trans(
            frames_feat, frames_mask,
            words_feat + words_pos, words_mask,
            decoding=1,
        )
        video_h = h[:, :n_frames]

        (scale_score, query_scale_feat, video_density_feat,
         v_global, _, q_global) = self.scale_estimator(video_h, words_feat, words_mask)

        gate_weights = self.gating_network(
            v_global, q_global, words_feat, words_mask,
            query_scale_feat, video_density_feat,
            scale_score, training=False,
        )
        half_n = self.num_experts // 2
        macro_ratio = gate_weights[:, :half_n].sum(dim=-1)

        return {
            'routing_weights': gate_weights,
            'scale_scores': scale_score,
            'macro_ratio': macro_ratio,
        }

    # ------------------------------------------------------------------
    # Utilities shared with CPL / CPL_MoE
    # ------------------------------------------------------------------

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


# ==============================================================================
# Helper utilities (duplicated from cpl.py / cpl_moe.py to keep this file
# self-contained without modifying existing files)
# ==============================================================================

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
    """Sinusoidal positional embeddings of any length."""

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
