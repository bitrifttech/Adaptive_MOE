"""Expert management system for the Adaptive MoE."""

import json
from pathlib import Path
from typing import Dict, Optional, TypeVar

from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from ..utils.config import ExpertConfig
from ..utils.logging import get_logger

# Type variable for generic model types
ModelType = TypeVar("ModelType", bound=PreTrainedModel)

logger = get_logger(__name__)


class ExpertManager:
    """Manages expert models for the Adaptive MoE system."""

    def __init__(self, config: ExpertConfig, base_model: PreTrainedModel):
        """Initialize the ExpertManager.

        Args:
            config: Expert configuration
            base_model: The base model to use for creating experts
        """
        self.config = config
        self.base_model = base_model
        self.experts: Dict[str, PreTrainedModel] = {}
        self.expert_metadata: Dict[str, dict] = {}

        # Create expert cache directory if it doesn't exist
        self.cache_dir = Path(config.expert_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing experts from cache
        self._load_cached_experts()

        logger.info(
            f"Initialized ExpertManager with {len(self.experts)} cached experts"
        )

    def _get_expert_path(self, expert_id: str) -> Path:
        """Get the filesystem path for an expert.

        Args:
            expert_id: Unique identifier for the expert

        Returns:
            Path to the expert directory
        """
        return self.cache_dir / expert_id

    def _load_cached_experts(self) -> None:
        """Load all cached experts from disk."""
        if not self.cache_dir.exists():
            return

        for expert_dir in self.cache_dir.iterdir():
            if not expert_dir.is_dir():
                continue

            expert_id = expert_dir.name
            metadata_path = expert_dir / "metadata.json"

            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    self.expert_metadata[expert_id] = metadata
                    logger.info(f"Found cached expert: {expert_id}")
                except Exception as e:
                    logger.error(f"Error loading metadata for expert {expert_id}: {e}")

    def create_expert(
        self,
        expert_id: str,
        adapter_config: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> PreTrainedModel:
        """Create a new expert model.

        Args:
            expert_id: Unique identifier for the expert
            adapter_config: Configuration for the LoRA adapter
            metadata: Additional metadata about the expert

        Returns:
            The created expert model
        """
        if expert_id in self.experts:
            logger.warning(
                f"Expert {expert_id} already exists, returning existing expert"
            )
            return self.experts[expert_id]

        # Default adapter config if not provided
        if adapter_config is None:
            adapter_config = {
                "r": self.config.lora_rank,  # PEFT uses 'r' for rank
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "target_modules": self.config.target_modules,
            }

        # Create a copy of the base model
        expert = self.base_model

        # Create LoRA config
        peft_config = LoraConfig(
            r=adapter_config["r"],
            lora_alpha=adapter_config["lora_alpha"],
            lora_dropout=adapter_config["lora_dropout"],
            target_modules=adapter_config["target_modules"],
            task_type="CAUSAL_LM",
        )

        # Convert to PEFT model with LoRA adapter
        expert = get_peft_model(expert, peft_config)

        # Store the expert
        self.experts[expert_id] = expert

        # Store metadata
        self.expert_metadata[expert_id] = metadata or {}
        self.expert_metadata[expert_id]["adapter_config"] = adapter_config

        # Save to cache
        self._save_expert(expert_id)

        logger.info(f"Created new expert: {expert_id}")
        return expert

    def get_expert(self, expert_id: str) -> Optional[PreTrainedModel]:
        """Get an expert model by ID.

        Args:
            expert_id: ID of the expert to retrieve

        Returns:
            The expert model, or None if not found
        """
        # Check if expert is in memory
        if expert_id in self.experts:
            return self.experts[expert_id]

        # Try to load from cache
        expert_path = self._get_expert_path(expert_id)
        if not expert_path.exists():
            return None

        try:
            # Import here to avoid circular imports
            from peft import PeftModel

            # Load the adapter using PEFT's from_pretrained
            expert = PeftModel.from_pretrained(
                self.base_model, str(expert_path), adapter_name=expert_id
            )

            # Load metadata
            metadata_path = expert_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.expert_metadata[expert_id] = json.load(f)

            # Add to memory
            self.experts[expert_id] = expert

            # Enforce memory limit
            self._enforce_memory_limit()

            logger.info(f"Loaded expert from cache: {expert_id}")
            return expert

        except Exception as e:
            logger.error(f"Error loading expert {expert_id}: {e}")
            return None

    def _save_expert(self, expert_id: str) -> None:
        """Save an expert to disk.

        Args:
            expert_id: ID of the expert to save
        """
        if expert_id not in self.experts:
            return

        expert_path = self._get_expert_path(expert_id)
        expert_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save adapter using PEFT's save_pretrained
            self.experts[expert_id].save_pretrained(str(expert_path))

            # Save metadata
            metadata_path = expert_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.expert_metadata.get(expert_id, {}), f)

        except Exception as e:
            logger.error(f"Error saving expert {expert_id}: {e}")

    def _enforce_memory_limit(self) -> None:
        """Ensure we don't exceed the maximum number of experts in memory."""
        if len(self.experts) <= self.config.max_experts_in_memory:
            return

        # Remove the least recently used expert
        # TODO: Implement more sophisticated caching strategy if needed
        expert_to_remove = next(iter(self.experts))
        self.experts.pop(expert_to_remove)
        logger.info(f"Removed expert {expert_to_remove} from memory due to cache limit")

    def list_experts(self) -> Dict[str, dict]:
        """List all available experts with their metadata.

        Returns:
            Dictionary mapping expert IDs to their metadata
        """
        return self.expert_metadata.copy()

    def delete_expert(self, expert_id: str) -> bool:
        """Delete an expert.

        Args:
            expert_id: ID of the expert to delete

        Returns:
            True if the expert was deleted, False otherwise
        """
        # Remove from memory
        if expert_id in self.experts:
            del self.experts[expert_id]

        # Remove from metadata
        if expert_id in self.expert_metadata:
            del self.expert_metadata[expert_id]

        # Remove from disk
        expert_path = self._get_expert_path(expert_id)
        if expert_path.exists():
            import shutil

            try:
                shutil.rmtree(expert_path)
                logger.info(f"Deleted expert: {expert_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting expert {expert_id}: {e}")
                return False

        return False
