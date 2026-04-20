from typing import Optional
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from src.models.base import BaseVLM, ModelOutput
from src.models.factory import register_model
from src.utils.parsing import parse_citations


# InternVL image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_internvl_transform(input_size: int = 448) -> T.Compose:
    """Build the InternVL image transform pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image: Image.Image, input_size: int = 448) -> torch.Tensor:
    """Preprocess a single PIL image for InternVL.

    Returns a tensor of shape (1, 3, input_size, input_size).
    """
    transform = build_internvl_transform(input_size)
    return transform(image).unsqueeze(0)


class InternVL3Model(BaseVLM):
    """InternVL3-4B model wrapper."""

    @property
    def short_name(self) -> str:
        return "internvl3"

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=str(self.device),
            trust_remote_code=True,
        )
        self.model.eval()
        # Store processor reference for base class compatibility
        self.processor = self.tokenizer

    def build_messages(self, prompt_text: str, image: Optional[Image.Image] = None) -> list[dict]:
        if image is not None:
            text = f"<image>\n{prompt_text}"
        else:
            text = prompt_text
        return [{"role": "user", "content": text}]

    def generate(self, prompt_text: str, image: Optional[Image.Image] = None) -> ModelOutput:
        messages = self.build_messages(prompt_text, image)
        question = messages[0]["content"]

        # Prepare pixel values if image is provided
        pixel_values = None
        if image is not None:
            pixel_values = preprocess_image(image).to(
                device=self.device, dtype=self.dtype
            )

        # InternVL3 uses model.chat() for inference
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_config["temperature"] = self.temperature

        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
        )

        # model.chat returns (response_text, history) or just response_text
        if isinstance(response, tuple):
            raw_text = response[0]
        else:
            raw_text = response
        raw_text = raw_text.strip()

        # Approximate token counts (InternVL's chat doesn't return them directly)
        input_tokens = len(self.tokenizer.encode(question))
        output_tokens = len(self.tokenizer.encode(raw_text))

        return ModelOutput(
            raw_text=raw_text,
            parsed_citations=parse_citations(raw_text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


register_model("internvl3", InternVL3Model, "OpenGVLab/InternVL3-4B")
