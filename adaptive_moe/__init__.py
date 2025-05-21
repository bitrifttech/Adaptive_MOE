"""Adaptive Mixture of Experts system with dynamic expert creation.

This package implements an adaptive mixture of experts system that can dynamically
create and manage domain-specific expert models based on incoming queries.
"""
from adaptive_moe.utils.config import AdaptiveMoEConfig, load_config
from adaptive_moe.utils.model_utils import generate_text, load_base_model

__version__ = "0.1.0"

__all__ = [
    "AdaptiveMoEConfig",
    "load_config",
    "load_base_model",
    "generate_text",
]
