"""Tests for the MultiExpertRouter implementation."""

import pytest
import torch

from adaptive_moe.router import MultiExpertRouter
from adaptive_moe.utils.config import RouterConfig


@pytest.fixture
def router_config():
    """Create a router configuration for testing."""
    return RouterConfig(
        hidden_size=64,
        expert_selection_threshold=0.3,
        max_experts_per_token=2,
        router_dropout=0.1,
        use_router_bias=True,
    )


class TestMultiExpertRouter:
    """Test suite for the MultiExpertRouter class."""

    def test_initialization(self, router_config):
        """Test router initialization."""
        num_experts = 4
        router = MultiExpertRouter(router_config, num_experts=num_experts)

        assert router.num_experts == num_experts
        assert router.hidden_size == router_config.hidden_size
        assert router.threshold == router_config.expert_selection_threshold
        assert router.max_experts_per_token == router_config.max_experts_per_token

        # Check router network architecture
        assert isinstance(router.router_network, torch.nn.Sequential)
        assert len(router.router_network) == 2  # Linear + Dropout
        assert router.router_network[0].in_features == router_config.hidden_size
        assert router.router_network[0].out_features == num_experts

    def test_forward_pass(self, router_config):
        """Test forward pass with basic input."""
        batch_size = 2
        seq_len = 5
        hidden_size = router_config.hidden_size
        num_experts = 3

        router = MultiExpertRouter(router_config, num_experts=num_experts)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Test forward pass
        outputs = router(hidden_states)

        # Check output shapes
        assert "dispatch_mask" in outputs
        assert "weights" in outputs
        assert "load_balancing_loss" in outputs
        assert "router_logits" in outputs

        assert outputs["dispatch_mask"].shape == (batch_size, seq_len, num_experts)
        assert outputs["weights"].shape == (batch_size, seq_len, num_experts)
        assert isinstance(outputs["load_balancing_loss"], torch.Tensor)
        assert outputs["router_logits"].shape == (batch_size, seq_len, num_experts)

        # Check that weights are normalized per token
        weights_sum = outputs["weights"].sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6)

    def test_expert_selection(self, router_config):
        """Test expert selection logic."""
        batch_size = 1
        seq_len = 10
        hidden_size = router_config.hidden_size
        num_experts = 4

        router = MultiExpertRouter(router_config, num_experts=num_experts)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Get expert selection
        selection = router.get_expert_selection(hidden_states)

        # Check output structure
        assert "selected_experts" in selection
        assert "expert_weights" in selection
        assert "metrics" in selection

        # Check shapes and types
        assert len(selection["selected_experts"]) == batch_size
        assert len(selection["selected_experts"][0]) == seq_len
        assert len(selection["expert_weights"]) == batch_size
        assert len(selection["expert_weights"][0]) == seq_len

        # Check that at most max_experts_per_token are selected
        for i in range(batch_size):
            for j in range(seq_len):
                assert (
                    len(selection["selected_experts"][i][j])
                    <= router_config.max_experts_per_token
                )
                assert len(selection["expert_weights"][i][j]) == len(
                    selection["selected_experts"][i][j]
                )

                # Check that weights sum to ~1 for each token
                if selection["expert_weights"][i][j]:
                    weight_sum = sum(selection["expert_weights"][i][j])
                    assert (
                        abs(weight_sum - 1.0) < 1e-5
                    ), f"Weights sum to {weight_sum}, expected ~1.0"

    def test_load_balancing_loss(self, router_config):
        """Test load balancing loss calculation."""
        batch_size = 4
        seq_len = 1
        hidden_size = router_config.hidden_size
        num_experts = 3

        router = MultiExpertRouter(router_config, num_experts=num_experts)

        # Create test inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Get router outputs
        outputs = router(hidden_states)

        # Check loss is a scalar
        assert outputs["load_balancing_loss"].dim() == 0

        # Check loss is in reasonable range
        assert 0 <= outputs["load_balancing_loss"].item() <= num_experts

    def test_threshold_behavior(self, router_config):
        """Test that threshold affects expert selection."""
        batch_size = 1
        seq_len = 1
        hidden_size = router_config.hidden_size
        num_experts = 4

        # Set a very high threshold - should select few experts
        high_threshold_config = RouterConfig(
            **{**router_config.to_dict(), "expert_selection_threshold": 0.9}
        )

        # Set a very low threshold - should select many experts
        low_threshold_config = RouterConfig(
            **{**router_config.to_dict(), "expert_selection_threshold": 0.1}
        )

        router_high = MultiExpertRouter(high_threshold_config, num_experts=num_experts)
        router_low = MultiExpertRouter(low_threshold_config, num_experts=num_experts)

        # Use the same input for both routers
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Get expert selections
        high_selection = router_high.get_expert_selection(hidden_states)
        low_selection = router_low.get_expert_selection(hidden_states)

        # On average, high threshold should select fewer experts than low threshold
        high_avg = (
            sum(len(experts[0]) for experts in high_selection["selected_experts"])
            / batch_size
        )
        low_avg = (
            sum(len(experts[0]) for experts in low_selection["selected_experts"])
            / batch_size
        )

        assert high_avg <= low_avg
