"""Tests for the MoE layer implementation."""

import pytest
import torch
import torch.nn as nn

from adaptive_moe.integration.moe_layer import MoELayer
from adaptive_moe.router import MultiExpertRouter
from adaptive_moe.utils.config import RouterConfig


class DummyExpert(nn.Module):
    """Dummy expert for testing purposes."""

    def __init__(self, hidden_size, expert_id=0):
        """Initialize the dummy expert with a linear layer.

        Args:
            hidden_size: Size of the input and output features
            expert_id: Unique identifier for the expert (used for weight initialization)
        """
        super().__init__()
        self.expert_id = expert_id
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize with different weights for each expert
        nn.init.constant_(self.linear.weight, expert_id + 1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """Forward pass through the expert.

        Args:
            x: Input tensor

        Returns:
            Transformed output tensor
        """
        return self.linear(x)


@pytest.fixture
def router_config():
    """Create a router configuration for testing."""
    return RouterConfig(
        hidden_size=32,
        expert_selection_threshold=0.1,  # Low threshold to ensure experts are selected
        max_experts_per_token=2,
        router_dropout=0.0,  # Disable dropout for deterministic tests
        use_router_bias=False,  # Disable bias for deterministic tests
    )


@pytest.fixture
def dummy_experts(router_config):
    """Create a list of dummy experts for testing."""
    num_experts = 4
    experts = nn.ModuleList(
        [
            DummyExpert(router_config.hidden_size, expert_id=i)
            for i in range(num_experts)
        ]
    )
    return experts


class TestMoELayer:
    """Test suite for the MoELayer class."""

    def test_initialization(self, router_config, dummy_experts):
        """Test that MoELayer initializes correctly."""
        """Test MoELayer initialization."""
        moe_layer = MoELayer(router_config, dummy_experts)

        assert moe_layer.num_experts == len(dummy_experts)
        assert isinstance(moe_layer.router, MultiExpertRouter)
        assert moe_layer.experts is dummy_experts

        # Check that all experts are on the same device
        device = next(moe_layer.parameters()).device
        for expert in moe_layer.experts:
            assert next(expert.parameters()).device == device

    def test_forward_pass(self, router_config, dummy_experts):
        """Test the forward pass through MoELayer."""
        """Test forward pass through the MoE layer."""
        batch_size = 2
        seq_len = 5
        hidden_size = router_config.hidden_size

        moe_layer = MoELayer(router_config, dummy_experts)

        # Create test input
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Forward pass
        outputs = moe_layer(hidden_states)

        # Check output structure
        assert "hidden_states" in outputs
        assert "router_outputs" in outputs
        assert "expert_outputs" in outputs
        assert "load_balancing_loss" in outputs

        # Check output shape
        assert outputs["hidden_states"].shape == (batch_size, seq_len, hidden_size)

        # Check that router outputs are present
        router_outputs = outputs["router_outputs"]
        assert "selected_experts" in router_outputs
        assert "expert_weights" in router_outputs

        # Check that expert outputs are captured
        assert len(outputs["expert_outputs"]) > 0

        # Check that load balancing loss is a scalar tensor
        if outputs["load_balancing_loss"] is not None:
            assert isinstance(outputs["load_balancing_loss"], torch.Tensor)
            assert outputs["load_balancing_loss"].dim() == 0

    def test_attention_mask(self, router_config, dummy_experts):
        """Test that attention mask is properly handled."""
        batch_size = 2
        seq_len = 5
        hidden_size = router_config.hidden_size

        moe_layer = MoELayer(router_config, dummy_experts)

        # Create test input with padding (attention_mask=0 for padding)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[0, -1] = 0  # Mask the last token of the first example
        attention_mask[1, -2:] = 0  # Mask the last two tokens of the second example

        # Forward pass with attention mask
        outputs = moe_layer(hidden_states, attention_mask=attention_mask)

        # Check that masked positions were zeroed out (since we're using attention mask)
        # The MoELayer should zero out outputs for masked positions
        assert torch.all(outputs["hidden_states"][0, -1] == 0)
        assert torch.all(outputs["hidden_states"][1, -2:] == 0)

        # Check that non-masked positions were modified
        assert not torch.allclose(
            outputs["hidden_states"][0, 0], hidden_states[0, 0], atol=1e-6
        )

    def test_load_balancing_loss(self, router_config, dummy_experts):
        """Test that load balancing loss is properly handled."""
        moe_layer = MoELayer(router_config, dummy_experts)

        # Get initial loss (should be zero)
        initial_loss = moe_layer.get_load_balancing_loss()

        # Do a forward pass to compute the loss
        batch_size = 2
        seq_len = 5
        hidden_size = router_config.hidden_size
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        moe_layer(hidden_states)

        # Get the loss after forward pass
        loss = moe_layer.get_load_balancing_loss()

        # Check that loss is a scalar tensor and has changed from initial value
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        if initial_loss is not None:
            assert not torch.allclose(loss, initial_loss)

    def test_to_device(self, router_config, dummy_experts):
        """Test moving MoELayer to a different device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device test")

        moe_layer = MoELayer(router_config, dummy_experts)

        # Move to CPU (explicitly, in case default is GPU)
        moe_layer = moe_layer.to("cpu")
        assert next(moe_layer.parameters()).device.type == "cpu"

        # Move to CUDA
        moe_layer = moe_layer.to("cuda")
        assert next(moe_layer.parameters()).device.type == "cuda"

        # Check that all experts were moved to CUDA
        for expert in moe_layer.experts:
            assert next(expert.parameters()).device.type == "cuda"

        # Check that router was moved to CUDA
        assert next(moe_layer.router.parameters()).device.type == "cuda"
