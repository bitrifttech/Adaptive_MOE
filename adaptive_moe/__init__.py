"""Adaptive Mixture of Experts system with dynamic expert creation.

This package implements an adaptive mixture of experts system that can dynamically
create and manage domain-specific expert models based on incoming queries.
"""

__version__ = "0.1.0"

# Import only the config module to avoid circular imports
from adaptive_moe.utils.config import AdaptiveMoEConfig, load_config

# Lazy imports for heavy dependencies
__all__ = [
    "AdaptiveMoEConfig",
    "load_config",
]


# Define lazy imports for other modules
def __getattr__(name):
    if name == "load_base_model" or name == "generate_text":
        from adaptive_moe.utils.model_utils import generate_text, load_base_model

        globals()[name] = globals()[name] = (
            load_base_model if name == "load_base_model" else generate_text
        )
        return globals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
