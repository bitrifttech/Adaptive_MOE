"""Tests for the expert manager component."""

import json
import tempfile
from pathlib import Path

import pytest

from adaptive_moe.experts.expert_manager import ExpertManager
from adaptive_moe.utils.config import ExpertConfig, ModelConfig
from adaptive_moe.utils.model_utils import load_base_model


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for testing expert caching."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def base_model():
    """Create a small base model for testing."""
    model_config = ModelConfig(
        base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="float16",
        load_in_4bit=False,
        load_in_8bit=False,
    )
    model, _ = load_base_model(model_config)
    return model


def test_expert_manager_initialization(temp_cache_dir, base_model):
    """Test that the ExpertManager initializes correctly."""
    config = ExpertConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        max_experts_in_memory=3,
        expert_cache_dir=str(temp_cache_dir),
    )

    manager = ExpertManager(config, base_model)

    assert manager is not None
    assert len(manager.list_experts()) == 0
    assert manager.cache_dir == temp_cache_dir


def test_create_expert(temp_cache_dir, base_model):
    """Test creating a new expert."""
    config = ExpertConfig(expert_cache_dir=str(temp_cache_dir))

    manager = ExpertManager(config, base_model)
    expert_id = "test_expert"

    # Create a new expert
    expert = manager.create_expert(expert_id)

    assert expert is not None
    assert expert_id in manager.list_experts()

    # Verify the expert was saved to disk
    expert_dir = temp_cache_dir / expert_id
    assert expert_dir.exists()

    # Check for files created by PEFT's save_pretrained
    assert (expert_dir / "adapter_config.json").exists()
    assert (expert_dir / "adapter_model.bin").exists() or (
        expert_dir / "adapter_model.safetensors"
    ).exists()

    # Check for metadata file
    assert (expert_dir / "metadata.json").exists()

    # Verify metadata content
    with open(expert_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    assert "adapter_config" in metadata
    assert metadata["adapter_config"]["r"] == config.lora_rank
    assert metadata["adapter_config"]["lora_alpha"] == config.lora_alpha


def test_get_expert(temp_cache_dir, base_model):
    """Test retrieving an expert."""
    config = ExpertConfig(expert_cache_dir=str(temp_cache_dir))

    manager = ExpertManager(config, base_model)
    expert_id = "test_expert"

    # Create and then retrieve the expert
    manager.create_expert(expert_id)
    expert = manager.get_expert(expert_id)

    assert expert is not None
    assert expert_id in manager.list_experts()


def test_memory_management(temp_cache_dir, base_model):
    """Test that the expert manager respects memory limits."""
    config = ExpertConfig(max_experts_in_memory=2, expert_cache_dir=str(temp_cache_dir))

    manager = ExpertManager(config, base_model)

    # Create more experts than the memory limit
    for i in range(3):
        manager.create_expert(f"expert_{i}")

    # All experts should be in memory because we're not enforcing the limit yet
    assert len(manager.experts) == 3

    # All experts should be listed in metadata
    assert len(manager.list_experts()) == 3

    # Try to get an expert which should trigger memory management
    manager.get_expert("expert_0")

    # Now we should have at most max_experts_in_memory + 1 in memory
    assert len(manager.experts) <= config.max_experts_in_memory + 1


def test_delete_expert(temp_cache_dir, base_model):
    """Test deleting an expert."""
    config = ExpertConfig(expert_cache_dir=str(temp_cache_dir))

    manager = ExpertManager(config, base_model)
    expert_id = "test_expert"

    # Create and then delete the expert
    _ = manager.create_expert(expert_id)
    assert expert_id in manager.list_experts()

    # Verify the expert directory exists
    expert_dir = temp_cache_dir / expert_id
    assert expert_dir.exists()

    # Delete the expert
    result = manager.delete_expert(expert_id)
    assert result is True

    # Verify the expert is no longer in the manager
    assert expert_id not in manager.list_experts()

    # Verify the expert directory was deleted
    assert not expert_dir.exists()

    # Verify we can't delete a non-existent expert
    assert manager.delete_expert("non_existent_expert") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
