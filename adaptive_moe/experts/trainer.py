"""Training pipeline for expert models in Adaptive MoE."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, TypeVar, Union

import torch
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from ..utils.config import ExpertConfig, ModelConfig, TrainerConfig
from ..utils.logging import get_logger
from .expert_manager import ExpertManager

logger = get_logger(__name__)
ModelType = TypeVar("ModelType", bound=PreTrainedModel)


@dataclass
class TrainingMetrics:
    """Container for tracking training metrics."""

    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    epoch: int = 0
    global_step: int = 0


class ExpertTrainer:
    """Handles training and fine-tuning of expert models."""

    def __init__(
        self,
        model_config: ModelConfig,
        expert_config: ExpertConfig,
        trainer_config: TrainerConfig,
        output_dir: Union[str, Path],
    ):
        """Initialize the expert trainer.

        Args:
            model_config: Configuration for the base model.
            expert_config: Configuration for expert-specific settings.
            trainer_config: Training hyperparameters and settings.
            output_dir: Directory to save checkpoints and logs.
        """
        self.model_config = model_config
        self.expert_config = expert_config
        self.trainer_config = trainer_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load base model and tokenizer
        self.model, self.tokenizer = self._load_base_model()
        self.model = self.model.to(self.device)

        # Set up expert manager
        self.expert_manager = ExpertManager(expert_config, self.model)

    def _load_base_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the base model and tokenizer."""
        from ..utils.model_utils import load_base_model  # Avoid circular import

        model, tokenizer = load_base_model(self.model_config)
        return model, tokenizer

    def _prepare_dataset(
        self, dataset_name: Union[str, Dataset], split: str = "train"
    ) -> Tuple[Dataset, DataLoader]:
        """Prepare the dataset for training.

        Args:
            dataset_name: Name, path of the dataset, or a Dataset object.
            split: Dataset split to use (train/validation). Only used if
                dataset_name is a string.

        Returns:
            Tuple of (dataset, dataloader).
        """
        # If dataset_name is already a Dataset, use it directly
        if isinstance(dataset_name, Dataset):
            dataset = dataset_name
        else:
            # Otherwise, try to load it as a path or dataset name
            try:
                # First try to load as a local directory
                if os.path.isdir(dataset_name):
                    dataset = load_from_disk(dataset_name)
                else:
                    # Otherwise, try to load from the hub
                    dataset = load_dataset(dataset_name, split=split)
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset from {dataset_name}. "
                    "Please provide a valid dataset path, name, or Dataset object."
                ) from e

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.trainer_config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Create dataloader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.trainer_config.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )

        return tokenized_dataset, dataloader

    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration for the model."""
        return LoraConfig(
            r=self.expert_config.lora_rank,
            lora_alpha=self.expert_config.lora_alpha,
            lora_dropout=self.expert_config.lora_dropout,
            target_modules=self.expert_config.target_modules,
            task_type="CAUSAL_LM",
        )

    def train(
        self,
        expert_id: str,
        train_dataset_name: str,
        eval_dataset_name: Optional[str] = None,
        num_epochs: Optional[int] = None,
    ) -> TrainingMetrics:
        """Train a new expert model.

        Args:
            expert_id: Unique identifier for the expert.
            train_dataset_name: Name or path of the training dataset.
            eval_dataset_name: Optional name or path of the evaluation dataset.
            num_epochs: Optional number of epochs to train for (overrides config).

        Returns:
            Training metrics.
        """
        # Set up training parameters
        num_epochs = num_epochs or self.trainer_config.num_train_epochs
        total_steps = (
            len(train_dataset_name)  # type: ignore
            * num_epochs
            // self.trainer_config.per_device_train_batch_size
        )
        warmup_steps = int(total_steps * self.trainer_config.warmup_ratio)

        # Prepare datasets
        train_dataset, train_dataloader = self._prepare_dataset(
            train_dataset_name, split="train"
        )
        eval_dataloader = None
        if eval_dataset_name:
            _, eval_dataloader = self._prepare_dataset(eval_dataset_name, split="test")

        # Create expert with LoRA adapter
        expert = self.expert_manager.create_expert(expert_id)
        expert.train()

        # Set up optimizer and scheduler
        optimizer = AdamW(
            expert.parameters(),
            lr=self.trainer_config.learning_rate,
            weight_decay=self.trainer_config.weight_decay,
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        metrics = TrainingMetrics()
        global_step = 0

        for epoch in range(num_epochs):
            metrics.epoch = epoch
            expert.train()
            total_loss = 0.0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = expert(**batch)
                loss = outputs.loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update metrics
                total_loss += loss.item()
                global_step += 1

                # Log metrics
                if global_step % self.trainer_config.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )

            # Calculate average loss for the epoch
            metrics.train_loss = total_loss / len(train_dataloader)
            metrics.learning_rate = lr_scheduler.get_last_lr()[0]
            metrics.global_step = global_step

            # Evaluate on validation set if available
            if eval_dataloader:
                metrics.eval_loss = self.evaluate(expert, eval_dataloader)
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={metrics.train_loss:.4f}, "
                    f"eval_loss={metrics.eval_loss:.4f}, "
                    f"lr={metrics.learning_rate:.2e}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={metrics.train_loss:.4f}, "
                    f"lr={metrics.learning_rate:.2e}"
                )

            # Save checkpoint
            if (epoch + 1) % self.trainer_config.save_steps == 0:
                self.save_checkpoint(expert_id, epoch + 1, metrics)

        # Save final model
        self.expert_manager._save_expert(expert_id)
        return metrics

    @torch.no_grad()
    def evaluate(self, model: PreTrainedModel, eval_dataloader: DataLoader) -> float:
        """Evaluate the model on the given dataloader.

        Args:
            model: Model to evaluate.
            eval_dataloader: Dataloader for evaluation.

        Returns:
            Average evaluation loss.
        """
        model.eval()
        total_loss = 0.0

        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

        return total_loss / len(eval_dataloader)

    def save_checkpoint(
        self, expert_id: str, epoch: int, metrics: TrainingMetrics
    ) -> None:
        """Save a training checkpoint.

        Args:
            expert_id: ID of the expert being trained.
            epoch: Current epoch number.
            metrics: Current training metrics.
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{expert_id}-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        expert = self.expert_manager.get_expert(expert_id)
        if expert:
            expert.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": metrics.global_step,
            "train_loss": metrics.train_loss,
            "eval_loss": metrics.eval_loss,
            "learning_rate": metrics.learning_rate,
        }

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(
        self, expert_id: str, checkpoint_dir: Union[str, Path]
    ) -> Tuple[PreTrainedModel, TrainingMetrics]:
        """Load a training checkpoint.

        Args:
            expert_id: ID of the expert to load.
            checkpoint_dir: Directory containing the checkpoint.

        Returns:
            Tuple of (model, metrics).
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load model
        expert = self.expert_manager.get_expert(expert_id)
        if expert is None:
            raise ValueError(f"Expert {expert_id} not found")

        # Load training state
        with open(checkpoint_dir / "training_state.json") as f:
            training_state = json.load(f)

        metrics = TrainingMetrics(**training_state)
        return expert, metrics
