"""Simple tests for the router trainer module."""

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_moe.router.router import MultiExpertRouter
from adaptive_moe.utils.config import RouterConfig


@pytest.fixture
def small_model():
    """Fixture providing a small pretrained model for testing."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    return model, tokenizer


@pytest.fixture
def test_dataset():
    """Create a small test dataset."""
    # Create a small dataset with random text
    texts = [f"This is test example {i} for the router training." for i in range(10)]
    return Dataset.from_dict({"text": texts})


def test_simple_router_forward(small_model, test_dataset):
    """Test a simple forward pass through the router."""
    model, _ = small_model

    # Create router config with hidden size from model
    router_config = RouterConfig(
        hidden_size=model.config.hidden_size,
        num_experts=2,
        expert_selection_threshold=0.1,
        max_experts_per_token=2,
    )

    # Initialize router directly
    router = MultiExpertRouter(router_config, num_experts=2)

    # Get a batch from the dataset
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize a small batch
    batch = tokenizer(
        ["This is a test", "Another test example"],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    # Get hidden states from base model
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-2]  # Second to last layer

        # Forward pass through router
        router_outputs = router(hidden_states)

        # Check output shapes
        batch_size, seq_len, _ = hidden_states.shape
        assert router_outputs["dispatch_mask"].shape == (
            batch_size,
            seq_len,
            2,
        )  # 2 experts
        assert router_outputs["weights"].shape == (batch_size, seq_len, 2)

        # Check that we have load balancing loss
        assert "load_balancing_loss" in router_outputs
        assert isinstance(router_outputs["load_balancing_loss"], torch.Tensor)
