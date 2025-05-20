"""Utility functions for model loading and management."""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import ModelConfig
from .logging import get_logger

logger = get_logger(__name__)


def load_base_model(
    config: ModelConfig, device_map: Optional[str] = None, **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the base language model and tokenizer.

    Args:
        config: Model configuration.
        device_map: Device map for model parallelism. Uses config if None.
        **kwargs: Additional keyword arguments to pass to from_pretrained.

    Returns:
        A tuple of (model, tokenizer).
    """
    logger.info(f"Loading base model: {config.base_model_id}")

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.base_model_id,
        padding_side="left",
        trust_remote_code=config.trust_remote_code,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up model kwargs
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
        "use_cache": config.use_cache,
        "device_map": device_map or config.device_map,
    }

    # Handle quantization
    if config.load_in_4bit or config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        # Set torch_dtype if not quantizing
        if hasattr(torch, config.torch_dtype):
            model_kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id, **model_kwargs, **kwargs
    )

    # Freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False

    # Set model to evaluation mode
    model.eval()

    logger.info(f"Successfully loaded model to {next(model.parameters()).device}")
    return model, tokenizer


def save_model(
    model: PreTrainedModel,
    save_dir: Union[str, Path],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs,
) -> None:
    """Save a model and optionally its tokenizer.

    Args:
        model: The model to save.
        save_dir: Directory to save the model to.
        tokenizer: Optional tokenizer to save with the model.
        **kwargs: Additional keyword arguments to pass to model.save_pretrained.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {save_dir}")
    model.save_pretrained(save_dir, **kwargs)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

    logger.info(f"Model successfully saved to {save_dir}")


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **generation_kwargs,
) -> str:
    """Generate text from the model.

    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        prompt: The input prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature for sampling.
        top_p: Nucleus sampling parameter.
        **generation_kwargs: Additional keyword arguments for generation.

    Returns:
        The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )

    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the generated part (after the input prompt)
    prompt_len = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    generated_text = full_response[prompt_len:]

    return generated_text.strip()
