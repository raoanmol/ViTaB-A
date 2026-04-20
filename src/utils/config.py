from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    experiment_name: str = "default"

    # Model
    model: str = "qwen3vl"
    model_name_override: Optional[str] = None

    # Data
    dataset: str = "hitab"
    split: str = "test"
    data_dir: str = "data"
    max_samples: Optional[int] = None

    # Table representation
    table_repr: str = "markdown"

    # Prompting
    prompt_strategy: str = "zero_shot"

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.0

    # Runtime
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 42

    # Output
    output_dir: str = "results"

    # Task type
    task: str = "inference"


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw_dict = yaml.safe_load(f) or {}
    return ExperimentConfig(**raw_dict)
