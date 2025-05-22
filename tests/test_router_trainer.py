"""Tests for the router trainer module."""

import tempfile

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_moe.router.router import MultiExpertRouter
from adaptive_moe.router.trainer import RouterTrainer
from adaptive_moe.utils.config import ModelConfig, RouterConfig, TrainerConfig


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


@pytest.fixture
def trainer_config():
    """Create a trainer configuration for testing."""
    return TrainerConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_steps=10,
        weight_decay=0.01,
        max_seq_length=32,
        logging_steps=1,
        save_steps=1,
    )


def test_router_trainer_init(small_model, trainer_config):
    """Test initialization of the router trainer."""
    model, _ = small_model

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model config with actual model name from fixture
        model_config = ModelConfig(
            base_model_id=model.config._name_or_path,
            torch_dtype="float32",
            device_map="auto",
        )

        # Create router config with hidden size from model
        router_config = RouterConfig(
            hidden_size=model.config.hidden_size,
            num_experts=2,
            expert_selection_threshold=0.1,
            max_experts_per_token=2,
        )

        # Initialize router directly
        router = MultiExpertRouter(router_config, num_experts=2)

        # Initialize trainer with the pre-initialized router
        trainer = RouterTrainer(
            model_config=model_config,
            router_config=router_config,
            trainer_config=trainer_config,
            output_dir=tmpdir,
            router=router,  # Pass the pre-initialized router
        )

        # Check that the model and tokenizer are properly set
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.router is not None
        assert trainer.device == torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


def test_router_trainer_forward_pass(small_model, test_dataset, trainer_config):
    """Test a single forward pass through the router trainer."""
    model, _ = small_model

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model config with actual model name from fixture
        model_config = ModelConfig(
            base_model_id=model.config._name_or_path,
            torch_dtype="float32",
            device_map="auto",
        )

        # Create router config with hidden size from model
        router_config = RouterConfig(
            hidden_size=model.config.hidden_size,
            num_experts=2,
            expert_selection_threshold=0.1,
            max_experts_per_token=2,
        )

        # Initialize router directly
        router = MultiExpertRouter(router_config, num_experts=2)

        # Initialize trainer with the pre-initialized router
        trainer = RouterTrainer(
            model_config=model_config,
            router_config=router_config,
            trainer_config=trainer_config,
            output_dir=tmpdir,
            router=router,  # Pass the pre-initialized router
        )

        # Get a batch from the dataset
        _, dataloader = trainer._prepare_dataset(test_dataset)
        batch = next(iter(dataloader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}

        # Forward pass through the model and router
        with torch.no_grad():
            # Get hidden states from base model
            outputs = trainer.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-2]  # Second to last layer

            # Forward pass through router
            router_outputs = trainer.router(hidden_states)

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


def test_router_trainer_evaluation(small_model, test_dataset, trainer_config):
    """Test the evaluation functionality of the router trainer."""
    model, _ = small_model

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model config with actual model name from fixture
        model_config = ModelConfig(
            base_model_id=model.config._name_or_path,
            torch_dtype="float32",
            device_map="auto",
        )

        # Create router config with hidden size from model
        router_config = RouterConfig(
            hidden_size=model.config.hidden_size,
            num_experts=2,
            expert_selection_threshold=0.1,
            max_experts_per_token=2,
        )

        # Initialize router directly
        router = MultiExpertRouter(router_config, num_experts=2)

        # Initialize trainer with the pre-initialized router
        trainer = RouterTrainer(
            model_config=model_config,
            router_config=router_config,
            trainer_config=trainer_config,
            output_dir=tmpdir,
            router=router,  # Pass the pre-initialized router
        )

        # Prepare the evaluation dataset and get the dataloader
        _, eval_dataloader = trainer._prepare_dataset(test_dataset, split="test")

        # Run evaluation without labels
        eval_metrics = trainer.evaluate(eval_dataloader)

        # Check that metrics were returned
        assert isinstance(eval_metrics.eval_loss, float)
        assert isinstance(eval_metrics.expert_utilization, dict)
        assert len(eval_metrics.expert_utilization) == 2  # Should have 2 experts
        assert eval_metrics.eval_loss >= 0  # Loss should be non-negative
