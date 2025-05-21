"""Configuration management for the Adaptive MoE system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml
from dataclasses_json import DataClassJsonMixin


@dataclass
class LoggingConfig(DataClassJsonMixin):
    """Configuration for logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"
    log_file: str = "adaptive_moe.log"


@dataclass
class ModelConfig(DataClassJsonMixin):
    """Configuration for model-related settings."""

    base_model_id: str = "NousResearch/Yarn-Mistral-7b-128k"
    tokenizer_name: Optional[str] = None
    torch_dtype: str = "float16"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    use_cache: bool = True


@dataclass
class RouterConfig(DataClassJsonMixin):
    """Configuration for the router component."""

    confidence_threshold: float = 0.7
    hidden_size: int = 4096  # Should match base model's hidden size
    router_learning_rate: float = 1e-5
    router_weight_decay: float = 0.01
    router_dropout: float = 0.1


@dataclass
class ExpertConfig(DataClassJsonMixin):
    """Configuration for expert models."""

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_experts_in_memory: int = 5
    expert_cache_dir: str = "expert_cache"


@dataclass
class TrainingConfig(DataClassJsonMixin):
    """Configuration for training settings."""

    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    max_grad_norm: float = 1.0
    max_steps: int = -1
    fp16: bool = True
    bf16: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    report_to: str = "wandb"


@dataclass
class AdaptiveMoEConfig(DataClassJsonMixin):
    """Main configuration class for the Adaptive MoE system."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "AdaptiveMoEConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            An instance of AdaptiveMoEConfig with values loaded from the file.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            config_path: Path where to save the YAML configuration file.
        """
        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> AdaptiveMoEConfig:
    """Load configuration from a file or return default configuration.

    Args:
        config_path: Optional path to a YAML configuration file.

    Returns:
        An instance of AdaptiveMoEConfig.
    """
    if config_path is not None and os.path.exists(config_path):
        return AdaptiveMoEConfig.from_yaml(config_path)
    return AdaptiveMoEConfig()
