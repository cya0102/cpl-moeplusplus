"""
Pilot Embedding Router for CPL-Router.

Implements RouterRetriever-style routing using learnable pilot embeddings.
Each expert maintains a set of pilot embeddings (centroids). Routing is
determined by cosine similarity between an input query embedding and the
pilot embeddings, averaged per expert.

Reference: ROUTERRETRIEVER: Routing over a Mixture of Expert Embedding Models (AAAI 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from models.modules.query_guided_moe import ProposalExpert


class PilotEmbeddingRouter(nn.Module):
    """
    RouterRetriever-style routing using learnable pilot embeddings.

    Each expert has `num_centroids` learnable pilot embeddings. Routing is
    determined by cosine similarity between the input query embedding
    and the pilot embeddings, averaged per expert.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_centroids: int = 1,
        top_k: int = 2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_centroids = num_centroids
        self.top_k = top_k
        self.temperature = temperature

        # Learnable pilot embeddings: [num_experts, num_centroids, hidden_size]
        self.pilot_embeddings = nn.Parameter(
            torch.empty(num_experts, num_centroids, hidden_size)
        )
        nn.init.xavier_uniform_(self.pilot_embeddings.view(-1, hidden_size))

        # Project fused (multimodal + query) features into pilot embedding space
        self.query_projector = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        multimodal_feat: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            multimodal_feat: [batch_size, hidden_size]
            query_feat: [batch_size, hidden_size]

        Returns:
            top_k_weights: [batch_size, top_k] re-normalized routing weights
            top_k_indices: [batch_size, top_k] selected expert indices
            routing_info: dict with 'expert_probs' for loss computation
        """
        bsz = multimodal_feat.size(0)

        # 1. Fuse and project into pilot embedding space
        fused = torch.cat([multimodal_feat, query_feat], dim=-1)
        routing_input = self.query_projector(fused)  # [bsz, hidden_size]

        # 2. L2 normalize routing input and pilot embeddings
        routing_input = F.normalize(routing_input, dim=-1)  # [bsz, D]
        pilots = F.normalize(self.pilot_embeddings, dim=-1)  # [E, C, D]

        # 3. Cosine similarity: [bsz, E, C]
        pilots_flat = pilots.view(-1, self.hidden_size)  # [E*C, D]
        similarities = torch.mm(routing_input, pilots_flat.t())  # [bsz, E*C]
        similarities = similarities.view(bsz, self.num_experts, self.num_centroids)

        # 4. Average over centroids per expert
        expert_scores = similarities.mean(dim=-1)  # [bsz, E]

        # 5. Temperature-scaled softmax
        expert_probs = F.softmax(expert_scores / self.temperature, dim=-1)  # [bsz, E]

        # 6. Top-k selection
        top_k_weights, top_k_indices = torch.topk(
            expert_probs, k=self.top_k, dim=-1
        )
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        routing_info = {
            'expert_probs': expert_probs,
        }

        return top_k_weights, top_k_indices, routing_info


class PilotRoutedMoE(nn.Module):
    """
    Mixture of Experts with pilot-embedding-based routing.
    Replaces fc_gauss in CPL for proposal parameter prediction.
    """
    def __init__(
        self,
        hidden_size: int,
        num_props: int,
        num_experts: int = 4,
        num_centroids: int = 1,
        top_k: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1,
        use_shared_expert: bool = True,
        use_load_balance_loss: bool = True,
        load_balance_weight: float = 0.01,
        use_diversity_loss: bool = True,
        diversity_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_props = num_props
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.use_load_balance_loss = use_load_balance_loss
        self.load_balance_weight = load_balance_weight
        self.use_diversity_loss = use_diversity_loss
        self.diversity_weight = diversity_weight

        # Pilot embedding router
        self.router = PilotEmbeddingRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_centroids=num_centroids,
            top_k=top_k,
            temperature=temperature,
        )

        # Expert networks (reuse ProposalExpert from query_guided_moe.py)
        self.experts = nn.ModuleList([
            ProposalExpert(hidden_size, num_props, dropout)
            for _ in range(num_experts)
        ])

        # Shared expert (always active, not routed)
        if use_shared_expert:
            self.shared_expert = ProposalExpert(hidden_size, num_props, dropout)
            self.output_gate = nn.Linear(num_props * 2 * 2, num_props * 2)

        # Layer normalization on input
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        multimodal_feat: torch.Tensor,
        query_feat: torch.Tensor,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            multimodal_feat: [batch_size, hidden_size]
            query_feat: [batch_size, hidden_size]
            return_aux_loss: whether to compute auxiliary losses

        Returns:
            gauss_params: [batch_size * num_props, 2] (center, width)
            aux_loss: scalar auxiliary loss or None
        """
        bsz = multimodal_feat.size(0)
        multimodal_feat = self.layer_norm(multimodal_feat)

        # 1. Route
        top_k_weights, top_k_indices, routing_info = self.router(
            multimodal_feat, query_feat
        )

        # 2. Compute routed expert outputs via top-k dispatch
        routed_output = torch.zeros(
            bsz, self.num_props * 2,
            device=multimodal_feat.device, dtype=multimodal_feat.dtype
        )

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]           # [bsz]
            expert_weights = top_k_weights[:, k:k + 1]     # [bsz, 1]

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = multimodal_feat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    routed_output[mask] += expert_weights[mask] * expert_output

        # 3. Fuse with shared expert
        if self.use_shared_expert:
            shared_output = self.shared_expert(multimodal_feat)
            combined = torch.cat([routed_output, shared_output], dim=-1)
            output = self.output_gate(combined)
        else:
            output = routed_output

        # 4. Sigmoid to get valid proposal params in [0, 1]
        gauss_params = torch.sigmoid(output).view(bsz * self.num_props, 2)

        # 5. Auxiliary losses
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = torch.tensor(0.0, device=multimodal_feat.device)
            if self.use_load_balance_loss:
                aux_loss = aux_loss + self._compute_load_balance_loss(
                    routing_info['expert_probs'], top_k_indices
                )
            if self.use_diversity_loss:
                aux_loss = aux_loss + self._compute_diversity_loss()

        return gauss_params, aux_loss

    def _compute_load_balance_loss(
        self,
        expert_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Switch Transformer style load balancing loss.
        Encourages balanced expert utilization.
        """
        bsz = expert_probs.size(0)

        # Fraction of tokens routed to each expert
        expert_counts = torch.zeros(
            self.num_experts, device=expert_probs.device
        )
        for k in range(self.top_k):
            for expert_id in range(self.num_experts):
                expert_counts[expert_id] += (top_k_indices[:, k] == expert_id).sum()
        tokens_per_expert = expert_counts / (bsz * self.top_k + 1e-6)

        # Mean routing probability per expert
        router_prob_per_expert = expert_probs.mean(dim=0)

        # Load balance loss
        loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()
        return self.load_balance_weight * loss

    def _compute_diversity_loss(self) -> torch.Tensor:
        """
        Penalize pilot embeddings from different experts being too similar.
        Encourages diverse, well-separated pilot embeddings.
        """
        pilots = F.normalize(self.router.pilot_embeddings, dim=-1)  # [E, C, D]
        E, C, D = pilots.shape

        # Flatten to [E*C, D]
        pilots_flat = pilots.view(-1, D)
        # Pairwise cosine similarity
        sim_matrix = torch.mm(pilots_flat, pilots_flat.t())  # [E*C, E*C]

        # Mask: 1 for cross-expert pairs, 0 for same-expert pairs
        expert_ids = torch.arange(E, device=pilots.device).unsqueeze(1).expand(E, C).reshape(-1)
        cross_mask = (expert_ids.unsqueeze(0) != expert_ids.unsqueeze(1)).float()

        # Mean cross-expert similarity (should be minimized)
        cross_sim = (sim_matrix * cross_mask).sum() / (cross_mask.sum() + 1e-6)
        return self.diversity_weight * cross_sim
