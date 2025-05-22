"""Tests for the router component."""

import pytest
import torch

from adaptive_moe.router.router import MultiExpertRouter
from adaptive_moe.utils.config import RouterConfig


def test_router_initialization():
    """Test that the router initializes correctly."""
    config = RouterConfig(
        hidden_size=768,
        expert_selection_threshold=0.3,
        max_experts_per_token=4,
        capacity_factor=1.25,
        use_router_bias=False,
        router_type="threshold",
    )
    num_experts = 4
    router = MultiExpertRouter(config, num_experts)

    assert router is not None
    assert isinstance(router, MultiExpertRouter)
    assert router.hidden_size == 768
    assert router.threshold == 0.3


def test_router_forward():
    """Test the forward pass of the router."""
    config = RouterConfig(
        hidden_size=768,
        expert_selection_threshold=0.3,
        max_experts_per_token=4,
        capacity_factor=1.25,
        use_router_bias=False,
        router_type="threshold",
    )
    num_experts = 4
    router = MultiExpertRouter(config, num_experts)

    # Create test input
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    router_output = router(hidden_states)

    # Check output contains expected keys
    assert "dispatch_mask" in router_output
    assert "weights" in router_output
    assert "load_balancing_loss" in router_output

    # Check output shapes
    assert router_output["dispatch_mask"].shape == (batch_size, seq_len, num_experts)
    assert router_output["weights"].shape == (batch_size, seq_len, num_experts)
    assert isinstance(router_output["load_balancing_loss"], torch.Tensor)


def test_router_batch_processing():
    """Test that the router handles batch processing correctly."""
    config = RouterConfig(
        hidden_size=768,
        expert_selection_threshold=0.3,
        max_experts_per_token=4,
        capacity_factor=1.25,
        use_router_bias=False,
        router_type="threshold",
    )
    num_experts = 4
    router = MultiExpertRouter(config, num_experts)

    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        hidden_states = torch.randn(batch_size, 10, 768)
        router_output = router(hidden_states)
        assert router_output["dispatch_mask"].shape == (batch_size, 10, num_experts)
        assert router_output["weights"].shape == (batch_size, 10, num_experts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
