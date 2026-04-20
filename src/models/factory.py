import torch
from src.models.base import BaseVLM

# Registry: short_name -> (class, default_hf_model_id)
# Individual model modules register themselves when imported
MODEL_REGISTRY: dict[str, tuple[type[BaseVLM], str]] = {}


def register_model(short_name: str, cls: type[BaseVLM], default_model_id: str) -> None:
    """Register a model class in the factory."""
    MODEL_REGISTRY[short_name] = (cls, default_model_id)


def create_model(name: str, device: torch.device, dtype: torch.dtype,
                 model_name: str | None = None, **kwargs) -> BaseVLM:
    """Create and load a model by its short name.

    Args:
        name: Short name from registry (e.g., "qwen3vl")
        device: torch device
        dtype: torch dtype
        model_name: Override the default HF model ID (optional)
        **kwargs: Forwarded to model constructor (max_new_tokens, temperature, etc.)

    Returns:
        Loaded BaseVLM instance ready for generate() calls.
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")

    cls, default_id = MODEL_REGISTRY[name]
    hf_id = model_name or default_id
    model = cls(model_name=hf_id, device=device, dtype=dtype, **kwargs)
    model.load()
    return model


def list_models() -> list[str]:
    """Return list of registered model short names."""
    return list(MODEL_REGISTRY.keys())
