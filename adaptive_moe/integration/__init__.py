"""Integration components for the Adaptive MoE system.

This module handles the integration of multiple experts with dynamic routing
and response combination.
"""

from .moe_layer import MoELayer

__all__ = ["MoELayer"]
