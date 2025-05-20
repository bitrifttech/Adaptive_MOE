"""Router implementation for the Adaptive MoE system."""

from typing import Tuple

import torch
import torch.nn as nn

from ..utils.config import RouterConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class UncertaintyRouter(nn.Module):
    """Router that directs queries based on model confidence."""

    def __init__(self, config: RouterConfig):
        """Initialize the UncertaintyRouter.

        Args:
            config: Router configuration.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.confidence_threshold = config.confidence_threshold

        # Simple feedforward network for confidence estimation
        self.estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.router_dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"Initialized UncertaintyRouter with hidden_size={self.hidden_size}, "
            f"threshold={self.confidence_threshold}"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Estimate confidence scores for the given hidden states.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor of shape (batch_size,) with confidence scores in [0, 1]
        """
        # Use mean pooling over sequence length
        pooled = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        return self.estimator(pooled).squeeze(-1)  # (batch_size,)

    def should_use_expert(self, hidden_states: torch.Tensor) -> Tuple[bool, float]:
        """Determine if an expert should be used based on confidence.

        Args:
            hidden_states: Tensor of shape (1, seq_len, hidden_size)

        Returns:
            Tuple of (use_expert: bool, confidence: float)
        """
        with torch.no_grad():
            confidence = self(hidden_states).item()
            return confidence < self.confidence_threshold, confidence
