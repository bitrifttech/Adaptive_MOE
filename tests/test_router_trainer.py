"""Tests for the router trainer module."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_moe.router.trainer import RouterTrainer
from adaptive_moe.utils.config import ModelConfig, RouterConfig, TrainerConfig


@pytest.fixture
def small_model():
    """Fixture providing a small pretrained model for testing."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_seq_length=32,
        logging_steps=1,
        save_steps=1,
    )


def test_router_trainer_init(small_model, trainer_config):
    """Test initialization of the router trainer."""
    model, tokenizer = small_model

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize trainer
        trainer = RouterTrainer(
            model_config=ModelConfig(
                base_model_id="test-model",
                hidden_size=model.config.hidden_size,
            ),
            router_config=RouterConfig(
                hidden_size=model.config.hidden_size,
                num_experts=4,
            ),
            trainer_config=trainer_config,
            output_dir=tmpdir,
        )

        # Check that the model and tokenizer are properly set
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.router is not None
        assert trainer.device == torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


def test_router_trainer_train_loop(small_model, test_dataset, trainer_config):
    """Test the training loop of the router trainer."""
    model, tokenizer = small_model

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize trainer with a smaller model and fewer epochs for testing
        trainer = RouterTrainer(
            model_config=ModelConfig(
                base_model_id="test-model",
                hidden_size=model.config.hidden_size,
            ),
            router_config=RouterConfig(
                hidden_size=model.config.hidden_size,
                num_experts=2,  # Fewer experts for testing
                expert_selection_threshold=0.1,  # Lower threshold for testing
            ),
            trainer_config=trainer_config,
            output_dir=tmpdir,
        )

        # Run training with the test dataset
        metrics = trainer.train(
            train_dataset=test_dataset,
            num_epochs=1,  # Just one epoch for testing
        )

        # Check that metrics were returned
        assert isinstance(metrics.train_loss, float)

        # Check that model files were saved
        output_dir = Path(tmpdir)
        assert (output_dir / "final_model" / "pytorch_model.bin").exists()
        assert (output_dir / "final_model" / "config.json").exists()


def test_router_trainer_evaluation(small_model, test_dataset, trainer_config):
    """Test the evaluation functionality of the router trainer."""
    model, tokenizer = small_model

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize trainer
        trainer = RouterTrainer(
            model_config=ModelConfig(
                base_model_id="test-model",
                hidden_size=model.config.hidden_size,
            ),
            router_config=RouterConfig(
                hidden_size=model.config.hidden_size,
                num_experts=2,
            ),
            trainer_config=trainer_config,
            output_dir=tmpdir,
        )

        # Run evaluation
        eval_metrics = trainer.evaluate(
            trainer._prepare_dataloader(test_dataset, is_train=False)
        )

        # Check that metrics were returned
        assert isinstance(eval_metrics.eval_loss, float)
        assert isinstance(eval_metrics.expert_utilization, dict)
        assert len(eval_metrics.expert_utilization) == 2  # Should have 2 experts


def test_router_trainer_save_load(small_model, test_dataset, trainer_config):
    """Test saving and loading of the router model."""
    model, tokenizer = small_model

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize trainer and train for one step
        trainer = RouterTrainer(
            model_config=ModelConfig(
                base_model_id="test-model",
                hidden_size=model.config.hidden_size,
            ),
            router_config=RouterConfig(
                hidden_size=model.config.hidden_size,
                num_experts=2,
            ),
            trainer_config=trainer_config,
            output_dir=tmpdir,
        )

        # Save the model
        save_dir = Path(tmpdir) / "saved_model"
        trainer.save_model(save_dir)

        # Check that files were saved
        assert (save_dir / "pytorch_model.bin").exists()
        assert (save_dir / "config.json").exists()

        # Create a new trainer and load the saved model
        new_trainer = RouterTrainer(
            model_config=ModelConfig(
                base_model_id="test-model",
                hidden_size=model.config.hidden_size,
            ),
            router_config=RouterConfig(
                hidden_size=model.config.hidden_size,
                num_experts=2,
            ),
            trainer_config=trainer_config,
            output_dir=tmpdir,
        )

        # Load the saved model
        new_trainer.load_model(save_dir)

        # Verify that the model was loaded correctly by running inference
        eval_metrics = new_trainer.evaluate(
            new_trainer._prepare_dataloader(test_dataset, is_train=False)
        )

        assert isinstance(eval_metrics.eval_loss, float)
