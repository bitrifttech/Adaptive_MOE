"""Router module for the Adaptive MoE system.

This module contains the implementation of the router that directs queries
to the most appropriate expert or falls back to the base model.
"""

from .router import UncertaintyRouter  # noqa: F401
