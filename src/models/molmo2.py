from typing import Optional
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from src.models.base import BaseVLM, ModelOutput
from src.models.factory import register_model
from src.utils.parsing import parse_citations


class Molmo2Model(BaseVLM):
    """Molmo-7B-D model wrapper."""

    @property
    def short_name(self) -> str:
        return "molmo2"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map=str(self.device),
        )
        self.model.eval()

    def build_messages(self, prompt_text: str, image: Optional[Image.Image] = None) -> list[dict]:
        # Molmo doesn't use structured chat messages
        # The processor handles image/text combination internally
        return [{"role": "user", "content": prompt_text}]

    def generate(self, prompt_text: str, image: Optional[Image.Image] = None) -> ModelOutput:
        # Molmo's processor.process() handles text + optional images
        images = [image] if image is not None else []
        inputs = self.processor.process(images=images, text=prompt_text)

        # Move inputs to device and add batch dimension
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        # Molmo uses generate_from_batch
        with torch.no_grad():
            output_ids = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    stop_strings=["<|endoftext|>"],
                ),
                tokenizer=self.processor.tokenizer,
            )

        # Decode only generated tokens
        generated_ids = output_ids[0, input_len:]
        raw_text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return ModelOutput(
            raw_text=raw_text,
            parsed_citations=parse_citations(raw_text),
            input_tokens=input_len,
            output_tokens=len(generated_ids),
        )

    def unload(self) -> None:
        """Override to handle Molmo's processor.tokenizer sub-attribute."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


register_model("molmo2", Molmo2Model, "allenai/Molmo-7B-D-0924")
