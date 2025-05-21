"""Main Adaptive MoE model implementation."""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..router import MultiExpertRouter
from ..utils.config import AdaptiveMoEConfig
from .moe_layer import MoELayer


@dataclass
class ModelOutput:
    """Simple model output container."""

    def __init__(self, **kwargs):
        """Initialize the ModelOutput with keyword arguments."""
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        """Get an attribute by key."""
        return getattr(self, key, None)

    def __contains__(self, key):
        """Check if key exists in the output."""
        return key in self.__dict__

    def keys(self):
        """Get all keys in the output."""
        return self.__dict__.keys()

    def items(self):
        """Get all key-value pairs in the output."""
        return self.__dict__.items()

    def __repr__(self):
        """Get string representation of the output."""
        return str(self.__dict__)


class AdaptiveMoE(nn.Module):
    """Adaptive Mixture of Experts model.

    This model integrates a base language model with multiple expert models
    using a dynamic routing mechanism.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,  # Simple tokenizer interface
        config: AdaptiveMoEConfig,
        expert_models: Optional[nn.ModuleList] = None,
    ):
        """Initialize the AdaptiveMoE model.

        Args:
            base_model: The base language model.
            tokenizer: Tokenizer for the base model.
            config: Configuration object.
            expert_models: Optional list of expert models. If None, will be initialized
                as copies of the base model.
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize experts if not provided
        if expert_models is None:
            # Create a simple config for the expert models if base model
            # doesn't have one
            expert_config = {}
            if hasattr(base_model, "config") and base_model.config is not None:
                expert_config = base_model.config
            elif hasattr(base_model, "hidden_size"):
                expert_config = {"hidden_size": base_model.hidden_size}

            expert_models = nn.ModuleList(
                [
                    type(base_model)(**expert_config)
                    for _ in range(config.router.num_experts)
                ]
            )
        self.expert_models = expert_models

        # Initialize router
        self.router = MultiExpertRouter(
            config=config.router, num_experts=len(expert_models)
        )

        # Get device from base model if available, otherwise use default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(base_model, "device") and base_model.device is not None:
            device = base_model.device

        # Get device and dtype from base model if available
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Initialize MoE layer with the same hidden size as base model
        self.moe_layer = MoELayer(
            router_config=config.router,
            experts=self.expert_models,
            device=device,
            dtype=dtype,
        )

        if hasattr(base_model, "device"):
            self.to(base_model.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            **kwargs: Additional arguments passed to the base model.

        Returns:
            Dictionary containing model outputs.
        """
        # Get base model hidden states
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Get hidden states before the final layer
        if (
            hasattr(base_outputs, "hidden_states")
            and base_outputs.hidden_states is not None
        ):
            hidden_states = base_outputs.hidden_states[-2]  # Second to last layer
        else:
            # If the model doesn't return hidden_states, use the last layer output
            hidden_states = (
                base_outputs.last_hidden_state
                if hasattr(base_outputs, "last_hidden_state")
                else None
            )

            if hidden_states is None:
                raise ValueError(
                    "Base model must return either 'hidden_states' or "
                    "'last_hidden_state' when output_hidden_states=True"
                )

        # Apply MoE layer
        moe_outputs = self.moe_layer(hidden_states, attention_mask)

        # Combine with base model outputs
        return {
            "last_hidden_state": moe_outputs["hidden_states"],
            "router_outputs": moe_outputs,
            "base_outputs": base_outputs,
        }

    def generate(self, *args, **kwargs):
        """Generate text using the base model's generation method."""
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model and tokenizer to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)

        # Save base model if it has a save_pretrained method
        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(save_directory, **kwargs)
        else:
            # Fallback to saving the state dict
            torch.save(
                self.base_model.state_dict(), f"{save_directory}/pytorch_model.bin"
            )

        # Save tokenizer if it has a save_pretrained method
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_directory)

        # Save expert models
        expert_dir = os.path.join(save_directory, "experts")
        os.makedirs(expert_dir, exist_ok=True)

        for i, expert in enumerate(self.expert_models):
            expert_path = os.path.join(expert_dir, f"expert_{i}")
            os.makedirs(expert_path, exist_ok=True)
            torch.save(
                expert.state_dict(), os.path.join(expert_path, "pytorch_model.bin")
            )

        # Save router and MoE layer
        torch.save(
            {
                "router_state_dict": self.router.state_dict(),
                "moe_layer_state_dict": self.moe_layer.state_dict(),
                "config": self.config.to_dict(),
            },
            os.path.join(save_directory, "moe_weights.bin"),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[AdaptiveMoEConfig] = None,
        **kwargs,
    ) -> "AdaptiveMoE":
        """Load a pretrained model from a directory or Hugging Face Hub.

        Args:
            model_name_or_path: Path to the model directory or model identifier from
                the Hugging Face Hub.
            config: Optional configuration object. If not provided, will try to load
                from the model directory.
            **kwargs: Additional keyword arguments passed to the base model's
                from_pretrained method.

        Returns:
            AdaptiveMoE: The loaded model.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config is None:
            # Try to load config from the model directory
            try:
                config = AdaptiveMoEConfig.from_pretrained(model_name_or_path)
            except OSError as e:
                raise ValueError(
                    f"Could not load config from {model_name_or_path}. "
                    "Please provide a config explicitly."
                ) from e

        # Load base model and tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize expert models
        expert_models = nn.ModuleList(
            [
                type(base_model)(base_model.config)
                for _ in range(config.router.num_experts)
            ]
        )

        # Initialize AdaptiveMoE
        model = cls(
            base_model=base_model,
            tokenizer=tokenizer,
            config=config,
            expert_models=expert_models,
        )

        # Load router and MoE layer weights if they exist
        moe_weights_path = os.path.join(model_name_or_path, "moe_weights.bin")
        if os.path.exists(moe_weights_path):
            weights = torch.load(moe_weights_path, map_location="cpu")
            model.router.load_state_dict(weights["router_state_dict"])
            model.moe_layer.load_state_dict(weights["moe_layer_state_dict"])

        return model
