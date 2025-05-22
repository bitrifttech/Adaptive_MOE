"""Tests for the expert trainer module."""

import os
from typing import Dict

import pytest
import torch
from datasets import Dataset, DatasetDict, load_from_disk

from adaptive_moe.experts.trainer import ExpertTrainer, TrainingMetrics
from adaptive_moe.utils.config import ExpertConfig, ModelConfig, TrainerConfig

# Use a smaller model for testing
TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def create_test_dataset(num_samples: int = 10, max_length: int = 64) -> DatasetDict:
    """Create a small test dataset for training.

    Args:
        num_samples: Number of samples in the dataset.
        max_length: Maximum sequence length.

    Returns:
        DatasetDict with train and test splits.
    """
    # Create DatasetDict with train and test splits
    test_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": ["test 1", "test 2", "test 3", "test 4"]}
            ),
            "test": Dataset.from_dict(
                {"text": ["eval 1", "eval 2", "eval 3", "eval 4"]}
            ),
        }
    )
    return test_dataset


class TestExpertTrainer:
    """Test suite for the ExpertTrainer class."""

    @pytest.fixture
    def trainer_config(self) -> TrainerConfig:
        """Create a trainer configuration for testing."""
        return TrainerConfig(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=1,
            logging_steps=1,
            gradient_accumulation_steps=1,
            save_steps=1,  # Save checkpoint after every step for testing
        )

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create a model configuration for testing."""
        return ModelConfig(
            base_model_id=TEST_MODEL_ID,
            torch_dtype="float32",
            load_in_4bit=False,
            load_in_8bit=False,
        )

    @pytest.fixture
    def expert_config(self) -> ExpertConfig:
        """Create an expert configuration for testing."""
        return ExpertConfig(
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            max_experts_in_memory=3,
            expert_cache_dir="./expert_cache",
        )

    @pytest.fixture
    def test_dataset(self) -> Dict:
        """Create a test dataset."""
        return create_test_dataset(num_samples=4)

    def test_trainer_initialization(
        self, trainer_config, model_config, expert_config, tmp_path
    ):
        """Test that the trainer initializes correctly."""
        trainer = ExpertTrainer(
            model_config=model_config,
            expert_config=expert_config,
            trainer_config=trainer_config,
            output_dir=tmp_path,
        )

        assert trainer is not None
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "tokenizer")
        assert hasattr(trainer, "device")
        assert trainer.device == torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def test_training_loop(
        self, trainer_config, model_config, expert_config, test_dataset, tmp_path
    ):
        """Test the training loop with a small dataset."""
        # Skip this test in CI to avoid downloading large models
        if os.environ.get("CI"):
            pytest.skip("Skipping training test in CI")

        # Create trainer
        trainer = ExpertTrainer(
            model_config=model_config,
            expert_config=expert_config,
            trainer_config=trainer_config,
            output_dir=str(tmp_path),  # Convert to string for compatibility
        )

        # Create a temporary directory for the test dataset
        dataset_path = str(tmp_path / "test_dataset")
        test_dataset.save_to_disk(dataset_path)

        try:
            # Load datasets from disk
            train_dataset = load_from_disk(os.path.join(dataset_path, "train"))
            eval_dataset = load_from_disk(os.path.join(dataset_path, "test"))

            # Train for one epoch
            expert_id = "test_expert"
            metrics = trainer.train(
                expert_id=expert_id,
                train_dataset_name=train_dataset,
                eval_dataset_name=eval_dataset,
                num_epochs=1,
            )

            # Verify metrics
            assert isinstance(metrics, TrainingMetrics)
            assert metrics.epoch == 0  # 0-based epoch counter
            assert metrics.train_loss > 0
            assert metrics.eval_loss is not None
            assert metrics.eval_loss > 0
            # Learning rate could be 0.0 if using linear schedule with 1 epoch
            # So we'll just check that it's a float
            assert isinstance(metrics.learning_rate, float)

            # Verify checkpoint was saved
            checkpoint_dir = tmp_path / f"checkpoint-{expert_id}-1"
            assert checkpoint_dir.exists()
            assert (checkpoint_dir / "adapter_config.json").exists()
            assert (checkpoint_dir / "adapter_model.safetensors").exists() or (
                checkpoint_dir / "adapter_model.bin"
            ).exists()
            assert (checkpoint_dir / "training_state.json").exists()

        finally:
            # Clean up
            if os.path.exists(dataset_path):
                import shutil

                shutil.rmtree(dataset_path)

    def test_evaluate_method(
        self, trainer_config, model_config, expert_config, test_dataset, tmp_path
    ):
        """Test the evaluate method."""
        # Skip this test in CI to avoid downloading large models
        if os.environ.get("CI"):
            pytest.skip("Skipping evaluation test in CI")

        # Create trainer
        trainer = ExpertTrainer(
            model_config=model_config,
            expert_config=expert_config,
            trainer_config=trainer_config,
            output_dir=str(tmp_path),  # Convert to string
        )

        # Create a temporary directory for the test dataset
        dataset_path = str(tmp_path / "test_dataset")  # Convert to string
        test_dataset.save_to_disk(dataset_path)

        try:
            # Load dataset from disk and prepare dataloader
            test_dataset = load_from_disk(os.path.join(dataset_path, "test"))
            _, eval_dataloader = trainer._prepare_dataset(test_dataset, split="test")

            # Evaluate
            eval_loss = trainer.evaluate(trainer.model, eval_dataloader)

            # Verify loss is a float
            assert isinstance(eval_loss, float)
            assert eval_loss > 0

        finally:
            # Clean up
            if os.path.exists(dataset_path):
                import shutil

                shutil.rmtree(dataset_path)

    def test_save_and_load_checkpoint(
        self, trainer_config, model_config, expert_config, test_dataset, tmp_path
    ):
        """Test saving and loading checkpoints."""
        # Skip this test in CI to avoid downloading large models
        if os.environ.get("CI"):
            pytest.skip("Skipping checkpoint test in CI")

        # Convert tmp_path to string for consistency
        tmp_path_str = str(tmp_path)

        # Create trainer
        trainer = ExpertTrainer(
            model_config=model_config,
            expert_config=expert_config,
            trainer_config=trainer_config,
            output_dir=tmp_path_str,
        )

        # Create a temporary directory for the test dataset
        dataset_path = os.path.join(tmp_path_str, "test_dataset")
        os.makedirs(dataset_path, exist_ok=True)
        test_dataset.save_to_disk(dataset_path)

        try:
            # Load datasets from disk
            train_dataset = load_from_disk(os.path.join(dataset_path, "train"))
            eval_dataset = load_from_disk(os.path.join(dataset_path, "test"))

            # Train for one epoch to create a checkpoint
            expert_id = "test_checkpoint"
            trainer.train(
                expert_id=expert_id,
                train_dataset_name=train_dataset,
                eval_dataset_name=eval_dataset,
                num_epochs=1,
            )

            # Define checkpoint directory
            checkpoint_dir = os.path.join(tmp_path_str, f"checkpoint-{expert_id}-1")

            # Load the checkpoint
            loaded_model, metrics = trainer.load_checkpoint(expert_id, checkpoint_dir)

            # Verify loaded model and metrics
            assert loaded_model is not None
            assert isinstance(metrics, TrainingMetrics)
            assert metrics.epoch == 1  # 1-based epoch counter
            assert metrics.train_loss > 0

        finally:
            # Clean up
            if os.path.exists(dataset_path):
                import shutil

                shutil.rmtree(dataset_path)
