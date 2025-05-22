#!/usr/bin/env python3
"""Script to train the router for the Adaptive MoE system."""

import argparse
import json
from pathlib import Path
from typing import Tuple

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from adaptive_moe.router.router import MultiExpertRouter
from adaptive_moe.router.trainer import RouterTrainer
from adaptive_moe.utils.config import AdaptiveMoEConfig, LoggingConfig, load_config
from adaptive_moe.utils.logging import setup_logging


def load_data(data_path: str) -> Tuple[Dataset, Dataset]:
    """Load training and evaluation datasets.

    Args:
        data_path: Path to the dataset directory

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    try:
        # Try to load from disk first
        train_path = Path(data_path) / "train"
        eval_path = Path(data_path) / "eval"

        if train_path.exists() and eval_path.exists():
            train_dataset = load_from_disk(str(train_path))
            eval_dataset = load_from_disk(str(eval_path))
            return train_dataset, eval_dataset

        # Fall back to loading from file
        data_path = Path(data_path)
        with open(data_path / "train.json", "r") as f:
            train_data = json.load(f)
        with open(data_path / "eval.json", "r") as f:
            eval_data = json.load(f)

        train_dataset = Dataset.from_dict(train_data)
        eval_dataset = Dataset.from_dict(eval_data)
        return train_dataset, eval_dataset

    except Exception as e:
        raise RuntimeError(f"Failed to load data from {data_path}: {str(e)}")


def main():
    """Run the router training pipeline."""
    parser = argparse.ArgumentParser(description="Train the router for Adaptive MoE")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Path to the training dataset directory",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Path to the evaluation dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/router_training",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_config = LoggingConfig(
        log_level="INFO",
        log_file=output_dir / "training.log",
        tensorboard_dir=output_dir / "tensorboard",
        wandb_project=None,
    )
    logger = setup_logging(logging_config)
    logger.info("Starting router training...")

    # Load configuration
    config = load_config(args.config, config_class=AdaptiveMoEConfig)

    # Override config with command line arguments
    if args.num_epochs is not None:
        config.training.num_train_epochs = args.num_epochs
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
        config.training.per_device_eval_batch_size = args.batch_size * 2

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)

    # Initialize router
    router = MultiExpertRouter(
        hidden_size=config.router.hidden_size,
        num_experts=config.router.num_experts,
        expert_selection_threshold=config.router.expert_selection_threshold,
        max_experts_per_token=config.router.max_experts_per_token,
        use_router_bias=config.router.use_router_bias,
        router_type=config.router.router_type,
        capacity_factor=config.router.capacity_factor,
    )

    # Initialize trainer
    trainer = RouterTrainer(
        router=router,
        tokenizer=tokenizer,
        trainer_config=config.training,
        output_dir=output_dir,
    )

    # Load datasets
    train_dataset, eval_dataset = load_data(args.train_dataset), load_data(
        args.eval_dataset
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=config.training.num_train_epochs,
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
