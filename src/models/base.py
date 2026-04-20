from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from PIL import Image
import torch


@dataclass
class ModelOutput:
    """Standardized output from any model."""
    raw_text: str                    # Full decoded model output
    parsed_citations: list[str]      # Extracted cell refs like ["=E7", "=B3"]
    input_tokens: int                # Number of input tokens
    output_tokens: int               # Number of generated tokens


class BaseVLM(ABC):
    """Abstract base class for all vision-language models."""

    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype,
                 max_new_tokens: int = 512, temperature: float = 0.0):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor/tokenizer onto device.
        Called once after construction, before any generate() calls."""
        ...

    @abstractmethod
    def build_messages(self, prompt_text: str, image: Optional[Image.Image] = None) -> list[dict]:
        """Convert prompt text and optional image into model-specific
        chat message format. Returns list of message dicts."""
        ...

    @abstractmethod
    def generate(self, prompt_text: str, image: Optional[Image.Image] = None) -> ModelOutput:
        """Run inference: format messages, tokenize, generate, decode, parse citations.
        This is the single public method the inference runner calls."""
        ...

    def unload(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short identifier for results directories (e.g., 'qwen3vl')."""
        ...
