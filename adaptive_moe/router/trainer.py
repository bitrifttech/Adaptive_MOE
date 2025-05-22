"""Training utilities for the router component of the Adaptive MoE system."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from adaptive_moe.router.router import MultiExpertRouter
from adaptive_moe.utils.config import TrainingConfig
from adaptive_moe.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouterTrainingMetrics:
    """Container for tracking router training metrics."""

    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    expert_utilization: Dict[int, float] = None  # Expert index to utilization rate
    load_balancing_loss: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    global_step: int = 0


class RouterTrainer:
    """Handles training of the router in the Adaptive MoE system."""

    def __init__(
        self,
        router: MultiExpertRouter,
        tokenizer: PreTrainedTokenizer,
        trainer_config: TrainingConfig,
        output_dir: Union[str, Path],
    ):
        """Initialize the router trainer.

        Args:
            router: The router model to train.
            tokenizer: Tokenizer for processing input data.
            trainer_config: Training hyperparameters and settings.
            output_dir: Directory to save checkpoints and logs.
        """
        self.router = router
        self.router_config = router.config  # Store router config for tokenization
        self.tokenizer = tokenizer
        self.trainer_config = trainer_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Move router to device
        self.router = self.router.to(self.device)

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

        # Tokenize the dataset with batched processing
        def tokenize_function(examples):
            # Tokenize the input text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.trainer_config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            # Generate random expert assignments for each token (0 to num_experts-1)
            # In a real scenario, you would use actual expert assignments
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids))

            # Create expert assignments with the same shape as input_ids
            expert_assignments = torch.randint(
                0,
                self.router_config.num_experts,
                input_ids.shape,
                device=input_ids.device,
            )

            # Set padding tokens to -100 (ignore index)
            expert_assignments = (
                expert_assignments * attention_mask - (1 - attention_mask) * 100
            )

            # Convert to lists for the dataset
            tokenized["labels"] = expert_assignments.tolist()
            return tokenized

        # Process the dataset using map with batched=True
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )

        # Convert the dataset to PyTorch tensors
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        # Simple collate function that returns batches as-is
        def collate_fn(examples):
            return {
                "input_ids": torch.stack([e["input_ids"] for e in examples]),
                "attention_mask": torch.stack([e["attention_mask"] for e in examples]),
                "labels": torch.stack([e["labels"] for e in examples]),
            }

        # Create dataloader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.trainer_config.per_device_train_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
        )

        return tokenized_dataset, dataloader

    def _compute_router_loss(
        self,
        router_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the router loss including load balancing.

        Args:
            router_outputs: Outputs from the router forward pass.
            labels: Optional target labels for supervised training.

        Returns:
            Total loss for the router.
        """
        # Get load balancing loss from router outputs
        load_balancing_loss = router_outputs.get("load_balancing_loss", 0.0)

        # Initialize classification loss to zero
        classification_loss = 0.0

        # Only compute classification loss if we have both labels and logits
        if labels is not None and "router_logits" in router_outputs:
            # Flatten batch and sequence dimensions for classification
            logits = router_outputs["router_logits"]
            logits_flat = logits.view(
                -1, logits.size(-1)
            )  # [batch*seq_len, num_experts]

            # If we have labels, use them for supervised training
            if labels is not None:
                labels_flat = labels.view(-1)  # [batch*seq_len]
                # Only compute loss for non-padding tokens (if padding is used)
                non_padding_mask = (labels_flat != -100) & (
                    labels_flat < self.router.num_experts
                )
                if non_padding_mask.any():
                    logits_flat = logits_flat[non_padding_mask]
                    labels_flat = labels_flat[non_padding_mask]
                    # Ensure we have valid labels to compute loss
                    if labels_flat.numel() > 0:
                        classification_loss = F.cross_entropy(
                            logits_flat, labels_flat, reduction="mean"
                        )

        # Combine losses with load balancing weight from config or default
        load_balancing_weight = getattr(
            self.router_config, "load_balancing_weight", 0.01
        )
        total_loss = classification_loss + (load_balancing_weight * load_balancing_loss)

        return total_loss

    def train(
        self,
        train_dataset: Union[str, Dataset],
        eval_dataset: Optional[Union[str, Dataset]] = None,
        num_epochs: Optional[int] = None,
    ) -> RouterTrainingMetrics:
        """Train the router model.

        Args:
            train_dataset: Training dataset (path, name, or Dataset object).
            eval_dataset: Optional evaluation dataset.
            num_epochs: Optional number of epochs to train for (overrides config).

        Returns:
            Training metrics.
        """
        # Set up training parameters
        num_epochs = num_epochs or self.trainer_config.num_train_epochs
        _, train_dataloader = self._prepare_dataset(train_dataset, split="train")
        _, eval_dataloader = (
            self._prepare_dataset(eval_dataset, split="validation")
            if eval_dataset is not None
            else (None, None)
        )

        # Calculate total training steps
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = self.trainer_config.warmup_steps

        # Set up optimizer and scheduler
        optimizer = AdamW(
            self.router.parameters(),
            lr=self.trainer_config.learning_rate,
            weight_decay=self.trainer_config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        global_step = 0
        best_eval_loss = float("inf")
        metrics = RouterTrainingMetrics()

        for epoch in range(num_epochs):
            self.router.train()
            total_loss = 0.0

            # Training epoch
            for batch in tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Training]",
            ):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get hidden states from base model
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-2]  # Second to last layer

                # Forward pass through router
                router_outputs = self.router(hidden_states)

                # Compute loss
                loss = self._compute_router_loss(
                    router_outputs, labels=batch.get("labels")
                )

                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.router.parameters(), self.trainer_config.max_grad_norm
                )

                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update metrics
                total_loss += loss.item()
                global_step += 1

                # Log training metrics
                if global_step % self.trainer_config.logging_steps == 0:
                    avg_train_loss = total_loss / self.trainer_config.logging_steps
                    logger.info(
                        f"Step {global_step}: Train Loss = {avg_train_loss:.4f}, "
                        f"LR = {scheduler.get_last_lr()[0]:.2e}"
                    )
                    total_loss = 0.0

            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info(
                    f"Epoch {epoch + 1} Evaluation - "
                    f"Eval Loss: {eval_metrics.eval_loss:.4f}, "
                    f"Load Balancing Loss: {eval_metrics.load_balancing_loss:.4f}"
                )

                # Save best model
                if eval_metrics.eval_loss < best_eval_loss:
                    best_eval_loss = eval_metrics.eval_loss
                    self.save_model(self.output_dir / "best_model")

            # Save checkpoint
            if (epoch + 1) % self.trainer_config.save_epochs == 0:
                checkpoint_dir = self.output_dir / f"checkpoint-{epoch + 1}"
                self.save_model(checkpoint_dir)

        return metrics

    def evaluate(self, eval_dataloader: DataLoader) -> RouterTrainingMetrics:
        """Evaluate the router on the given dataset.

        Args:
            eval_dataloader: DataLoader for evaluation data.

        Returns:
            Evaluation metrics.
        """
        self.router.eval()
        total_loss = 0.0
        total_samples = 0
        expert_utilization = {i: 0 for i in range(self.router_config.num_experts)}

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch["input_ids"].size(0)

                # Get hidden states from the batch (assuming they were precomputed)
                if "hidden_states" in batch:
                    hidden_states = batch["hidden_states"]
                else:
                    # If hidden states not provided, use random ones for testing
                    seq_length = batch["input_ids"].size(1)
                    hidden_size = self.router_config.hidden_size
                    hidden_states = torch.randn(
                        batch_size, seq_length, hidden_size, device=self.device
                    )

                # Forward pass through router
                router_outputs = self.router(hidden_states)

                # Compute loss without labels (only load balancing loss)
                loss = self._compute_router_loss(router_outputs, labels=None)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Track expert utilization
                if "dispatch_mask" in router_outputs:
                    # Count how many tokens are assigned to each expert
                    dispatch_mask = router_outputs["dispatch_mask"]
                    # Ensure we're working with a 2D tensor [num_tokens, num_experts]
                    if dispatch_mask.dim() > 2:
                        dispatch_mask = dispatch_mask.view(-1, dispatch_mask.size(-1))
                    expert_counts = dispatch_mask.sum(dim=0).cpu()
                    for i, count in enumerate(expert_counts):
                        expert_utilization[i] += count.item()

        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Calculate expert utilization percentages
        total_tokens = sum(expert_utilization.values()) or 1  # Avoid division by zero
        expert_utilization = {
            i: count / total_tokens for i, count in expert_utilization.items()
        }

        metrics = RouterTrainingMetrics()
        metrics.eval_loss = avg_loss
        metrics.expert_utilization = expert_utilization
        metrics.load_balancing_loss = router_outputs.get(
            "load_balancing_loss", 0.0
        ).item()

        return metrics

    def save_model(self, output_dir: Union[str, Path]) -> None:
        """Save the router model and configuration.

        Args:
            output_dir: Directory to save the model to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = output_dir / "pytorch_model.bin"
        torch.save(self.router.state_dict(), model_path)

        # Save configuration
        config = {
            "model_config": self.model_config.to_dict(),
            "router_config": self.router_config.to_dict(),
            "trainer_config": self.trainer_config.to_dict(),
        }
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {output_dir}")

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained router model.

        Args:
            model_path: Path to the saved model directory.
        """
        model_path = Path(model_path)

        # Load model weights
        weights_path = model_path / "pytorch_model.bin"
        if not weights_path.exists():
            weights_path = (
                model_path  # Handle case where model_path points directly to weights
            )

        state_dict = torch.load(weights_path, map_location=self.device)
        self.router.load_state_dict(state_dict)
        logger.info(f"Model loaded from {model_path}")
