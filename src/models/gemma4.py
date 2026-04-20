from typing import Optional
from PIL import Image
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

from src.models.base import BaseVLM, ModelOutput
from src.models.factory import register_model
from src.utils.parsing import parse_citations


class Gemma4Model(BaseVLM):
    """Gemma 3 4B-IT model wrapper."""

    @property
    def short_name(self) -> str:
        return "gemma4"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=str(self.device),
        )
        self.model.eval()

    def build_messages(self, prompt_text: str, image: Optional[Image.Image] = None) -> list[dict]:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def generate(self, prompt_text: str, image: Optional[Image.Image] = None) -> ModelOutput:
        messages = self.build_messages(prompt_text, image)

        # Gemma3's processor can tokenize and handle images in one call
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        generated_ids = output_ids[:, input_len:]
        raw_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return ModelOutput(
            raw_text=raw_text,
            parsed_citations=parse_citations(raw_text),
            input_tokens=input_len,
            output_tokens=generated_ids.shape[1],
        )


register_model("gemma4", Gemma4Model, "google/gemma-3-4b-it")
