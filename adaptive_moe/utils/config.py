"""Configuration management for the Adaptive MoE system."""

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
    """Configuration for the router component.

    Attributes:
        expert_selection_threshold: Minimum confidence score to consider an expert
        hidden_size: Dimensionality of the input features (should match base model)
        router_learning_rate: Learning rate for router parameters (default: 1e-5)
        router_weight_decay: Weight decay for router parameters (default: 0.01)
        router_dropout: Dropout probability for the router network
        max_experts_per_token: Max experts per token (default: 2)
        capacity_factor: Factor for expert capacity calculation
        router_type: Type of router to use ('threshold' or 'topk')
        use_router_bias: Whether to include bias terms in router layers
    """

    expert_selection_threshold: float = 0.3
    hidden_size: int = 4096  # Should match base model's hidden size
    router_learning_rate: float = 1e-5
    router_weight_decay: float = 0.01
    router_dropout: float = 0.1
    max_experts_per_token: int = 4
    capacity_factor: float = 1.25
    router_type: str = "threshold"  # 'threshold' or 'topk'
    use_router_bias: bool = True
    num_experts: int = 4  # Number of expert models to use


@dataclass
class ExpertConfig(DataClassJsonMixin):
    """Configuration for expert models."""

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_experts_in_memory: int = 5
    expert_cache_dir: str = ".cache/expert_cache"


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
    max_seq_length: int = 128  # Maximum sequence length for tokenization
    save_epochs: int = 1  # Save a checkpoint every N epochs
    fp16: bool = True
    bf16: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    report_to: str = "wandb"


@dataclass
class TrainerConfig(DataClassJsonMixin):
    """Configuration for the expert trainer.

    This configuration is specifically designed for the ExpertTrainer class
    and includes settings for both training and evaluation.
    """

    # Expert training configuration
    per_device_train_batch_size: int = 8  # Batch size per device for training
    per_device_eval_batch_size: int = 8  # Batch size per device for evaluation
    num_train_epochs: int = 3  # Number of training epochs
    learning_rate: float = 5e-5  # Learning rate for training
    weight_decay: float = 0.01  # Weight decay for optimization
    warmup_steps: int = 0  # Number of warmup steps for learning rate scheduler
    logging_steps: int = 10  # Log every X updates steps
    evaluation_strategy: str = "epoch"  # Evaluation strategy to adopt
    save_strategy: str = "epoch"  # The checkpoint save strategy to adopt
    load_best_model_at_end: bool = True  # Whether to load the best model found
    metric_for_best_model: str = "eval_loss"  # The metric to compare models
    greater_is_better: bool = False  # Whether better models have higher metric
    early_stopping_patience: int = 3  # Eval calls without improvement to stop
    early_stopping_threshold: float = 0.0  # Stop when metric doesn't improve by this
    gradient_accumulation_steps: int = 1  # Steps to accumulate before backward pass
    # The scheduler type to use (linear, cosine, cosine_with_restarts,
    # polynomial, constant, constant_with_warmup)
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1  # Ratio of total training steps for warmup
    max_seq_length: int = 128  # Maximum sequence length for tokenization
    save_steps: int = 500  # Save checkpoint every X updates steps

    def __post_init__(self):
        """Validate configuration values."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.per_device_train_batch_size > 0, "Batch size must be positive"
        assert self.warmup_steps >= 0, "Warmup steps must be non-negative"
        assert 0 <= self.warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
        assert self.max_seq_length > 0, "Maximum sequence length must be positive"


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
    if config_path is None:
        return AdaptiveMoEConfig()

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return AdaptiveMoEConfig.from_dict(config_dict)


def save_config(config: AdaptiveMoEConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration object to save.
        output_path: Path where to save the YAML configuration file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
