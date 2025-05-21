"""Integration module for Adaptive MoE.

This module handles the integration of multiple experts with dynamic routing
and response combination.
"""

from .model import AdaptiveMoE, ModelOutput
from .moe_layer import MoELayer

__all__ = ["AdaptiveMoE", "ModelOutput", "MoELayer"]
