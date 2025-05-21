"""Router module for the Adaptive MoE system.

This module contains the implementation of the router that directs queries
to the most appropriate expert(s) based on threshold-based gating.
"""

from .router import MultiExpertRouter  # noqa: F401

__all__ = ["MultiExpertRouter"]
