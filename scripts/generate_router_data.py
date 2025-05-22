#!/usr/bin/env python3
"""Script to generate synthetic training data for the router model."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for router model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/router_train",
        help="Directory to save the generated dataset",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10000,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4,
        help="Number of experts to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def generate_synthetic_data(num_examples, max_length, num_experts, seed=42):
    """Generate synthetic training data for the router model.

    This function creates a dataset with random token IDs and expert assignments.
    The expert assignments are generated randomly to simulate routing behavior.

    Args:
        num_examples: Number of examples to generate
        max_length: Maximum sequence length
        num_experts: Number of experts to simulate
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing the generated dataset
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate random token IDs (vocab size = 1000)
    input_ids = np.random.randint(1, 1000, size=(num_examples, max_length))

    # Generate attention masks (all ones for now)
    attention_mask = np.ones_like(input_ids)

    # Generate random expert assignments (one per token)
    # This simulates which expert would be assigned to each token
    expert_assignments = np.random.randint(
        0, num_experts, size=(num_examples, max_length)
    )

    # Create dataset
    dataset = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "expert_assignments": expert_assignments.tolist(),
    }

    return dataset


def main():
    """Generate and save synthetic dataset for router training."""
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print(f"Generating {args.num_examples} examples...")
    dataset = generate_synthetic_data(
        num_examples=args.num_examples,
        max_length=args.max_length,
        num_experts=args.num_experts,
        seed=args.seed,
    )

    # Save dataset
    output_path = output_dir / "dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
