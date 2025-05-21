"""Mixture of Experts (MoE) layer implementation with dynamic routing.

This module implements a PyTorch module that applies a Mixture of Experts pattern
with dynamic routing using the MultiExpertRouter.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..router import MultiExpertRouter
from ..utils.config import RouterConfig


class MoELayer(nn.Module):
    """A Mixture of Experts (MoE) layer with dynamic routing.

        This layer takes an input tensor and routes different parts of it to different
    expert models based on the MultiExpertRouter's decisions.

        Args:
            router_config: Configuration for the router
            experts: ModuleList of expert models
            device: Device to run the layer on
            dtype: Data type for the layer
    """

    def __init__(
        self,
        router_config: RouterConfig,
        experts: nn.ModuleList,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize the MoELayer.

        Args:
            router_config: Configuration for the router
            experts: ModuleList of expert models
            device: Device to run the layer on
            dtype: Data type for the layer
        """
        super().__init__()
        self.config = router_config
        self.num_experts = len(experts)
        self.experts = experts

        # Initialize the router with the number of experts
        self.router = MultiExpertRouter(router_config, num_experts=self.num_experts)

        # Store device and dtype for later use
        self.device = device
        self.dtype = dtype

        # Move to device if specified
        if device is not None:
            self.to(device, dtype=dtype)

        # Ensure the router and experts are on the same device
        self.device = next(self.router.parameters(), torch.tensor(0)).device
        self.to(self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the MoE layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Dictionary containing:
                - hidden_states: Output tensor after applying MoE
                - router_outputs: Outputs from the router
                - expert_outputs: List of expert outputs (for analysis)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get router outputs
        router_outputs = self.router.get_expert_selection(hidden_states)

        # Initialize output tensor
        final_output = torch.zeros_like(hidden_states)

        # Get expert selection and weights from router outputs
        selected_experts = router_outputs["selected_experts"]  # List[List[List[int]]]
        expert_weights = router_outputs["expert_weights"]  # List[List[List[float]]]

        # Store metrics for potential logging
        self.last_router_metrics = router_outputs.get("metrics", {})

        # Initialize list to store expert outputs for analysis
        expert_outputs = []

        # Process each example in the batch
        for i in range(batch_size):
            # Initialize output for this example
            example_output = torch.zeros_like(hidden_states[i])

            # Process each position in the sequence
            for j in range(seq_len):
                # Skip padding tokens if attention mask is provided
                if attention_mask is not None and attention_mask[i, j] == 0:
                    continue

                # Get selected experts and their weights for this token
                selected = selected_experts[i][j]
                weights = expert_weights[i][j]

                # If no experts are selected, pass through the original hidden state
                if not selected:
                    example_output[j] = hidden_states[i, j]
                    continue

                # Convert weights to a tensor if they aren't already
                if not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(
                        weights, device=hidden_states.device, dtype=hidden_states.dtype
                    )

                # Normalize weights to sum to 1
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights = weights / weight_sum

                # Get the current token's hidden state
                token_hidden = hidden_states[i, j].unsqueeze(0)  # (1, hidden_size)

                # Apply each selected expert and accumulate the weighted outputs
                token_output = torch.zeros_like(token_hidden)
                for expert_idx, weight in zip(selected, weights):
                    expert = self.experts[expert_idx]
                    expert_out = expert(token_hidden)
                    token_output += weight * expert_out

                # Store the weighted sum of expert outputs
                example_output[j] = token_output.squeeze(0)

                # Store expert outputs for analysis
                if i == 0:  # Only store for first example to save memory
                    expert_outputs.append(
                        {
                            "position": j,
                            "selected_experts": selected,
                            "weights": (
                                weights.tolist()
                                if torch.is_tensor(weights)
                                else weights
                            ),
                            "output": token_output.detach().cpu(),
                        }
                    )

            # Add this example's output to the final output
            final_output[i] = example_output

        return {
            "hidden_states": final_output,
            "router_outputs": router_outputs,
            "expert_outputs": expert_outputs,
            "load_balancing_loss": router_outputs.get("load_balancing_loss", None),
        }

    def get_load_balancing_loss(self) -> Optional[torch.Tensor]:
        """Get the load balancing loss from the router.

        Returns:
            The load balancing loss tensor if available, otherwise None.
        """
        if (
            hasattr(self, "last_router_metrics")
            and "load_balancing_loss" in self.last_router_metrics
        ):
            return torch.tensor(
                self.last_router_metrics["load_balancing_loss"],
                device=next(self.parameters()).device,
            )
        return None

    def to(self, device, *args, **kwargs):
        """Move the MoE layer to the specified device."""
        self.device = device
        self.router = self.router.to(device)
        self.experts = self.experts.to(device)
        return super().to(device, *args, **kwargs)
