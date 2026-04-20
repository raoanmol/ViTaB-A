import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

VALID_DATASETS = {"hitab", "fetaqa"}
VALID_SPLITS = {"train", "validation", "test"}


@dataclass
class ViTaBSample:
    """Single sample from the ViTaB-A dataset."""
    id: str
    split: str
    question: str
    answer: Union[list, str]
    citation: list[str]
    table_json: dict
    table_md: str
    table_images: dict[str, str]
    source: str
    source_id: str


def load_dataset(data_dir: str, dataset: str, split: str,
                 max_samples: Optional[int] = None) -> list[ViTaBSample]:
    """Load samples from local JSONL files.

    Args:
        data_dir: Root data directory (e.g., "data")
        dataset: "hitab" or "fetaqa"
        split: "train", "validation", or "test"
        max_samples: If set, stop reading after this many samples.

    Returns:
        List of ViTaBSample objects.

    Raises:
        FileNotFoundError: If the JSONL file doesn't exist.
        ValueError: If dataset or split name is invalid.
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset!r}. Must be one of {VALID_DATASETS}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split!r}. Must be one of {VALID_SPLITS}")

    path = Path(data_dir) / dataset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: list[ViTaBSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(ViTaBSample(**record))
            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples
