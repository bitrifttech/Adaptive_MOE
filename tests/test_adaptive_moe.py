"""Tests for the AdaptiveMoE model integration."""

import os

import pytest
import torch
import torch.nn as nn

from adaptive_moe.integration.model import AdaptiveMoE, ModelOutput
from adaptive_moe.utils.config import AdaptiveMoEConfig, RouterConfig


class SimpleModel(nn.Module):
    """Simple model for testing the AdaptiveMoE integration."""

    def __init__(
        self, hidden_size: int = 32, num_layers: int = 2, vocab_size: int = 1000
    ):
        """Initialize the SimpleModel.

        Args:
            hidden_size: Size of the hidden layers
            num_layers: Number of hidden layers
            vocab_size: Size of the vocabulary
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Initialize embedding layer with padding_idx=0
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,  # Use 0 for padding
        )

        # Initialize model layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.output = nn.Linear(hidden_size, vocab_size)

        # For tracking hidden states if needed
        self.hidden_states = []

    def forward(
        self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs
    ):
        """Perform forward pass for the SimpleModel.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (unused in this simple model)
            output_hidden_states: Whether to return all hidden states
            **kwargs: Additional arguments (unused)

        Returns:
            ModelOutput with logits and optional hidden states
        """
        # Convert input to long if it's not already
        if not isinstance(input_ids, torch.LongTensor):
            input_ids = input_ids.long()

        # Get embeddings
        hidden = self.embedding(input_ids)

        # Initialize hidden states list if needed
        all_hidden_states = []
        if output_hidden_states:
            all_hidden_states.append(hidden)

        # Pass through layers
        for layer in self.layers:
            hidden = layer(hidden)
            if output_hidden_states:
                all_hidden_states.append(hidden)

        # Get logits
        logits = self.output(hidden)

        # Create output dictionary with required fields
        output_data = {"logits": logits, "last_hidden_state": hidden}
        if output_hidden_states:
            output_data["hidden_states"] = torch.stack(all_hidden_states)

        return ModelOutput(**output_data)

    def generate(self, input_ids, max_length=20, **kwargs):
        """Generate output tokens for testing.

        This is a simple implementation that repeats the input tokens
        for testing purposes.
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.new_zeros((batch_size, max_length), dtype=torch.long)
        generated[:, : input_ids.shape[1]] = input_ids

        # Simple generation: just repeat the input tokens
        for i in range(input_ids.shape[1], max_length):
            if i >= input_ids.shape[1]:
                # If we've run out of input tokens, just repeat the last token
                generated[:, i] = input_ids[:, -1] if input_ids.size(1) > 0 else 0

            # For testing, we'll just return the input tokens repeated
            if i >= input_ids.size(1) * 2:
                break

        return generated


class SimpleTokenizer:
    """Simple tokenizer for testing purposes.

    This tokenizer provides basic token ID functionality needed for testing
    the AdaptiveMoE model without requiring a full tokenizer implementation.
    """

    def __init__(self):
        """Initialize the SimpleTokenizer with default token IDs."""
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors=None, **kwargs):
        """Tokenize the input text.

        Args:
            text: Input text to tokenize
            return_tensors: If "pt", return PyTorch tensors
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary containing input_ids and attention_mask
        """
        # Simple tokenization: split on spaces and convert to IDs
        words = text.split()
        input_ids = [
            hash(word) % 1000 + 2 for word in words
        ]  # 0 and 1 are special tokens

        if return_tensors == "pt":
            input_ids = torch.tensor(
                [input_ids], dtype=torch.long
            )  # Ensure long type for embedding

        attention_mask = (
            torch.ones_like(input_ids)
            if return_tensors == "pt"
            else [1] * len(input_ids)
        )
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.dtype != torch.long
        ):
            attention_mask = attention_mask.long()

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text, **kwargs):
        """Encode the input text to token IDs.

        Args:
            text: Input text to encode
            **kwargs: Additional arguments passed to __call__

        Returns:
            List or tensor of token IDs
        """
        return self(text, **kwargs)["input_ids"]

    def decode(self, token_ids, **kwargs):
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments (unused)

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return f"Generated text with {len(token_ids)} tokens"

    def save_pretrained(self, save_directory):
        """Save the tokenizer to a directory.

        Args:
            save_directory: Directory to save the tokenizer to
        """
        os.makedirs(save_directory, exist_ok=True)


class TestAdaptiveMoE:
    """Test suite for the AdaptiveMoE model."""

    @pytest.fixture
    def model_config(self):
        """Create a test configuration.

        Returns:
            AdaptiveMoEConfig: Test configuration for AdaptiveMoE model.
        """
        return AdaptiveMoEConfig(
            router=RouterConfig(
                hidden_size=32,
                num_experts=4,
                expert_selection_threshold=0.1,
                max_experts_per_token=2,
                router_learning_rate=1e-5,
                router_weight_decay=0.01,
                router_dropout=0.1,
                capacity_factor=1.25,
                router_type="threshold",
                use_router_bias=True,
            )
        )

    @pytest.fixture
    def base_model_and_tokenizer(self):
        """Create a simple base model and tokenizer for testing.

        Returns:
            tuple: A tuple containing (base_model, tokenizer) for testing.
        """
        model = SimpleModel(hidden_size=32, num_layers=2)
        tokenizer = SimpleTokenizer()
        return model, tokenizer

    def test_initialization(self, model_config, base_model_and_tokenizer):
        """Test model initialization.

        Args:
            model_config: Fixture providing the model configuration.
            base_model_and_tokenizer: Fixture providing base model and tokenizer.
        """
        base_model, tokenizer = base_model_and_tokenizer
        model = AdaptiveMoE(base_model, tokenizer, model_config)
        assert model is not None
        assert len(model.expert_models) == model_config.router.num_experts
        assert model.router is not None
        assert model.moe_layer is not None

    def test_forward_pass(self, model_config, base_model_and_tokenizer):
        """Test the forward pass of the model.

        Args:
            model_config: Fixture providing the model configuration.
            base_model_and_tokenizer: Fixture providing base model and tokenizer.
        """
        base_model, tokenizer = base_model_and_tokenizer
        model = AdaptiveMoE(base_model, tokenizer, model_config)

        # Create test input
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)

        # Check output shapes
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (
            1,
            input_ids.shape[1],
            model_config.router.hidden_size,
        )
        assert "router_outputs" in outputs
        assert "base_outputs" in outputs

    def test_save_and_load(self, tmp_path, model_config, base_model_and_tokenizer):
        """Test saving and loading the model.

        Args:
            tmp_path: Fixture providing a temporary path.
            model_config: Fixture providing the model configuration.
            base_model_and_tokenizer: Fixture providing base model and tokenizer.
        """
        base_model, tokenizer = base_model_and_tokenizer
        model = AdaptiveMoE(base_model, tokenizer, model_config)

        # Save and load
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)

        # Create a new model and load the state
        new_model = AdaptiveMoE(base_model, tokenizer, model_config)
        new_model.load_state_dict(torch.load(save_path))

        # Check that the models have the same parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_generate(self, model_config, base_model_and_tokenizer):
        """Test text generation.

        Args:
            model_config: Fixture providing the model configuration.
            base_model_and_tokenizer: Fixture providing base model and tokenizer.
        """
        base_model, tokenizer = base_model_and_tokenizer
        model = AdaptiveMoE(base_model, tokenizer, model_config)

        # Test generation
        input_text = "Hello world"
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=20,
                num_return_sequences=1,
                do_sample=False,  # Disable sampling for deterministic tests
                pad_token_id=tokenizer.pad_token_id,
            )

        # Verify output
        assert outputs.shape[0] == 1  # Batch size 1
        assert outputs.shape[1] >= input_ids.shape[1]  # At least as long as input

        # Just verify we can decode without errors
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert isinstance(decoded, str)
