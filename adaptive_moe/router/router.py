"""Router implementation for Adaptive MoE with threshold-based expert selection."""

from typing import Dict

import torch
import torch.nn as nn

from ..utils.config import RouterConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultiExpertRouter(nn.Module):
    """Router that selects multiple experts per token.

    This router can select 0 or more experts per input token based on
    learned gating values. It implements the following features:
    - Threshold-based expert selection
    - Load balancing between experts
    - Support for variable number of experts
    - Efficient computation with capacity factor
    """

    def __init__(self, config: RouterConfig, num_experts: int):
        """Initialize the router.

        Args:
            config: Router configuration
            num_experts: Number of available experts.
        """
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size
        self.threshold = config.expert_selection_threshold
        self.capacity_factor = config.capacity_factor
        self.max_experts_per_token = config.max_experts_per_token

        # Router network that produces logits for each expert
        self.router_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, num_experts),
        )

        # Initialize router weights
        self._init_weights()

        logger.info(
            f"Initialized MultiExpertRouter with {num_experts} experts, "
            f"threshold={self.threshold}, max_experts={self.max_experts_per_token}"
        )

    def _init_weights(self):
        """Initialize router weights with small random values."""
        nn.init.normal_(
            self.router_network[0].weight, mean=0.0, std=1e-2 / self.num_experts
        )
        if self.config.use_router_bias:
            nn.init.constant_(self.router_network[0].bias, 0.0)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the router.

        Args:
            hidden_states: Input tensor of shape
                [batch_size, seq_len, hidden_size]

        Returns:
            Dictionary containing expert routing information
        """
        batch_size, seq_len, _ = hidden_states.shape
        _ = batch_size * seq_len  # Unused, kept for reference

        # Flatten batch and sequence dimensions
        hidden_states_flat = hidden_states.view(
            -1, self.hidden_size
        )  # [num_tokens, hidden_size]

        # Get router logits for each expert
        router_logits = self.router_network(
            hidden_states_flat
        )  # [num_tokens, num_experts]

        # Calculate gating values with sigmoid
        expert_gates = torch.sigmoid(router_logits)  # [num_tokens, num_experts]

        # Apply threshold to get expert selection mask
        expert_mask = expert_gates > self.threshold  # [num_tokens, num_experts]

        # Limit number of experts per token
        if self.max_experts_per_token > 0:
            top_k = min(self.max_experts_per_token, self.num_experts)
            top_k_gates, top_k_indices = torch.topk(
                expert_gates * expert_mask.float(), k=top_k, dim=1
            )
            expert_mask = torch.zeros_like(expert_gates, dtype=torch.bool)
            expert_mask.scatter_(1, top_k_indices, top_k_gates > 0)

        # Calculate load balancing loss
        _ = expert_mask.float().sum(0)  # [num_experts], unused
        router_probs = torch.softmax(router_logits, dim=1)
        load_balancing_loss = self._load_balancing_loss(
            router_probs, expert_mask.float()
        )

        # Calculate dispatch weights (normalized gating values for selected experts)
        weights = expert_gates * expert_mask.float()
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        normalized_weights = weights / weights_sum

        # Reshape back to original dimensions
        expert_mask = expert_mask.view(batch_size, seq_len, self.num_experts)
        normalized_weights = normalized_weights.view(
            batch_size, seq_len, self.num_experts
        )

        return {
            "dispatch_mask": expert_mask,
            "weights": normalized_weights,
            "load_balancing_loss": load_balancing_loss,
            "router_logits": router_logits.view(batch_size, seq_len, self.num_experts),
        }

    def _load_balancing_loss(
        self, router_probs: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate load balancing loss to encourage uniform expert utilization."""
        # Calculate fraction of tokens assigned to each expert
        routing_frac = expert_mask.mean(0)  # [num_experts]

        # Calculate average router probability for each expert
        router_frac = router_probs.mean(0)  # [num_experts]

        # Load balancing loss is the dot product of these fractions
        load_balancing_loss = torch.sum(routing_frac * router_frac) * self.num_experts

        return load_balancing_loss

    def get_expert_selection(
        self, hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get expert selection for the given hidden states.

        This is a convenience method that calls forward() and processes the results
        for easier integration with the rest of the system.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Dictionary containing:
            - selected_experts: List of lists of expert indices for each token
            - expert_weights: List of lists of weights for each selected expert
            - metrics: Dictionary with routing metrics
        """
        with torch.no_grad():
            router_output = self.forward(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        expert_mask = router_output[
            "dispatch_mask"
        ]  # [batch_size, seq_len, num_experts]
        expert_weights = router_output["weights"]  # [batch_size, seq_len, num_experts]

        # Convert to lists of selected experts and their weights
        selected_experts = []
        weights_list = []

        for i in range(batch_size):
            batch_experts = []
            batch_weights = []

            for j in range(seq_len):
                # Get indices of selected experts for this token
                expert_indices = torch.where(expert_mask[i, j])[0].tolist()
                batch_experts.append(expert_indices)

                # Get corresponding weights
                token_weights = expert_weights[i, j, expert_indices].tolist()
                batch_weights.append(token_weights)

            selected_experts.append(batch_experts)
            weights_list.append(batch_weights)

        # Calculate routing metrics
        num_selected = expert_mask.sum(dim=2).float().mean().item()
        router_confidence = expert_weights.max(dim=2)[0].mean().item()

        metrics = {
            "num_selected_experts": num_selected,
            "router_confidence": router_confidence,
            "load_balancing_loss": router_output["load_balancing_loss"].item(),
        }

        return {
            "selected_experts": selected_experts,
            "expert_weights": weights_list,
            "metrics": metrics,
        }
