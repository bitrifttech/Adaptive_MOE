"""Tests for the router component."""
import pytest
import torch

from adaptive_moe.router.router import UncertaintyRouter
from adaptive_moe.utils.config import RouterConfig


def test_router_initialization():
    """Test that the router initializes correctly."""
    config = RouterConfig(hidden_size=768, confidence_threshold=0.7, router_dropout=0.1)
    router = UncertaintyRouter(config)

    assert router is not None
    assert isinstance(router, UncertaintyRouter)
    assert router.hidden_size == 768
    assert router.confidence_threshold == 0.7


def test_router_forward():
    """Test the forward pass of the router."""
    config = RouterConfig(hidden_size=768, confidence_threshold=0.7, router_dropout=0.1)
    router = UncertaintyRouter(config)

    # Create test input
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    confidence_scores = router(hidden_states)

    # Check output shape and values
    assert confidence_scores.shape == (batch_size,)
    assert torch.all(confidence_scores >= 0.0) and torch.all(confidence_scores <= 1.0)


def test_should_use_expert():
    """Test the should_use_expert method."""
    config = RouterConfig(hidden_size=768, confidence_threshold=0.7, router_dropout=0.1)
    router = UncertaintyRouter(config)

    # Create test input
    seq_len = 10
    hidden_size = 768
    hidden_states = torch.randn(1, seq_len, hidden_size)

    # Test should_use_expert
    use_expert, confidence = router.should_use_expert(hidden_states)

    # Check return types and values
    assert isinstance(use_expert, bool)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

    # Verify threshold logic
    if confidence < 0.7:
        assert use_expert is True
    else:
        assert use_expert is False


def test_router_batch_processing():
    """Test that the router handles batch processing correctly."""
    config = RouterConfig(hidden_size=768, confidence_threshold=0.7, router_dropout=0.1)
    router = UncertaintyRouter(config)

    # Create batch input
    batch_sizes = [1, 2, 4]
    seq_len = 10
    hidden_size = 768

    for batch_size in batch_sizes:
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        confidence_scores = router(hidden_states)
        assert confidence_scores.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
