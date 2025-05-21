"""Test script to verify Yarn Mistral 7B model loading."""

import torch

from adaptive_moe.utils.config import AdaptiveMoEConfig
from adaptive_moe.utils.model_utils import load_base_model


def test_yarn_mistral_loading():
    """Test loading a smaller model for testing purposes."""
    print("Testing TinyLlama model loading...")

    # Create a minimal config for testing
    from adaptive_moe.utils.config import ModelConfig

    # Using TinyLlama for testing - much smaller and faster to load
    model_config = ModelConfig(
        base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_8bit=False,  # Disable quantization for testing
        load_in_4bit=False,  # Ensure 4-bit is also disabled
        torch_dtype="float16",
    )
    config = AdaptiveMoEConfig(model=model_config)

    print("Loading model and tokenizer...")
    try:
        model, tokenizer = load_base_model(config.model)
        print("✅ Successfully loaded Yarn Mistral 7B model and tokenizer")

        # Test a simple inference
        print("\nTesting model inference...")
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✅ Successfully ran inference. Output shape: {outputs.logits.shape}")
        print(f"✅ Model device: {next(model.parameters()).device}")

        # Check model configuration for context length
        if hasattr(model.config, "max_position_embeddings"):
            print(
                f"✅ Model context length: {model.config.max_position_embeddings} tokens"
            )

        return True

    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        raise


if __name__ == "__main__":
    test_yarn_mistral_loading()
