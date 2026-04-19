from typing import Any, Tuple

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

_ModelAndProcessor = Tuple[Any, Any]


def _create_gemma4_model(hf_id: str, _torch_dtype: str, _use_flash_attention: bool) -> _ModelAndProcessor:
    processor = AutoProcessor.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype="auto",
        device_map="auto",
    ).eval()
    return model, processor


def _create_internvl_model(hf_id: str, _torch_dtype: str, use_flash_attention: bool) -> _ModelAndProcessor:
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attention,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    return model, tokenizer


def _create_qwen_model(hf_id: str, _torch_dtype: str, use_flash_attention: bool) -> _ModelAndProcessor:
    processor = AutoProcessor.from_pretrained(hf_id)
    kwargs: dict = {"device_map": "auto"}
    if use_flash_attention:
        kwargs["dtype"] = torch.bfloat16
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["dtype"] = "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(hf_id, **kwargs).eval()
    return model, processor


def _create_molmo2_model(hf_id: str, _torch_dtype: str, _use_flash_attention: bool) -> _ModelAndProcessor:
    processor = AutoProcessor.from_pretrained(
        hf_id,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, processor


MODEL_REGISTRY = {
    # Gemma 4 family
    "gemma4_e4b": {
        "hf_id": "google/gemma-4-E4B-it",
        "create": _create_gemma4_model,
    },
    "gemma4_26b": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "create": _create_gemma4_model,
    },
    "gemma4_31b": {
        "hf_id": "google/gemma-4-31B-it",
        "create": _create_gemma4_model,
    },
    # InternVL 3.5 family
    "internvl35_4b": {
        "hf_id": "OpenGVLab/InternVL3_5-4B-hf",
        "create": _create_internvl_model,
    },
    "internvl35_8b": {
        "hf_id": "OpenGVLab/InternVL3_5-8B-hf",
        "create": _create_internvl_model,
    },
    "internvl35_14b": {
        "hf_id": "OpenGVLab/InternVL3_5-14B-hf",
        "create": _create_internvl_model,
    },
    "internvl35_38b": {
        "hf_id": "OpenGVLab/InternVL3_5-38B-hf",
        "create": _create_internvl_model,
    },
    # Qwen3-VL family
    "qwen3_vl_2b": {
        "hf_id": "Qwen/Qwen3-VL-2B-Instruct",
        "create": _create_qwen_model,
    },
    "qwen3_vl_4b": {
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "create": _create_qwen_model,
    },
    "qwen3_vl_8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "create": _create_qwen_model,
    },
    "qwen3_vl_32b": {
        "hf_id": "Qwen/Qwen3-VL-32B-Instruct",
        "create": _create_qwen_model,
    },
    # Molmo2 family
    "molmo2_4b": {
        "hf_id": "allenai/Molmo2-4B",
        "create": _create_molmo2_model,
    },
    "molmo2_8b": {
        "hf_id": "allenai/Molmo2-8B",
        "create": _create_molmo2_model,
    },
}


def create_model(model_name: str, torch_dtype: str = "bfloat16", use_flash_attention: bool = False) -> _ModelAndProcessor:
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")

    entry = MODEL_REGISTRY[model_name]
    return entry["create"](entry["hf_id"], torch_dtype, use_flash_attention)
