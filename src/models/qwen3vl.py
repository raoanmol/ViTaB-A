from typing import Optional
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.models.base import BaseVLM, ModelOutput
from src.models.factory import register_model
from src.utils.parsing import parse_citations


class Qwen3VLModel(BaseVLM):
    """Qwen3-VL model wrapper."""

    @property
    def short_name(self) -> str:
        return "qwen3vl"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        # Apply chat template
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info (extracts PIL images from message dicts)
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize
        inputs = self.processor(
            text=[text_input],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        # Decode only generated tokens
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


# Register with factory
register_model("qwen3vl", Qwen3VLModel, "Qwen/Qwen3-VL-4B-Instruct")
