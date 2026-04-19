---
dataset_info:
  - config_name: hitab
    data_files:
      - split: train
        path: hitab/train.jsonl
      - split: validation
        path: hitab/validation.jsonl
      - split: test
        path: hitab/test.jsonl
  - config_name: fetaqa
    data_files:
      - split: train
        path: fetaqa/train.jsonl
      - split: validation
        path: fetaqa/validation.jsonl
      - split: test
        path: fetaqa/test.jsonl
---

# ViTaB-A Dataset

A normalized table question answering dataset for the ViTaB-A research project.

## Configs

- **hitab**: Derived from [HiTab](https://huggingface.co/datasets/kasnerz/hitab) (10,670 samples)
- **fetaqa**: Derived from [FeTaQA](https://huggingface.co/datasets/DongfuJiang/FeTaQA) (10,330 samples)

## Usage

```python
from datasets import load_dataset

# Load a specific config
hitab = load_dataset("path/to/ViTaB-A", "hitab")
fetaqa = load_dataset("path/to/ViTaB-A", "fetaqa")
```

## Schema

Each sample contains:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier (e.g. `vitaba_000001_hitab`) |
| `split` | string | Dataset split (train/validation/test) |
| `question` | string | Natural language question about the table |
| `answer` | list or string | Answer (list for HiTab, string for FeTaQA) |
| `citation` | list[str] | Excel-style cell references (e.g. `["=E7"]`) |
| `table_json` | dict | Simplified table with keys: title (string), header (list of header rows), rows (list of data rows) |
| `table_md` | string | Markdown representation of the table with Excel-style row/column labels |
| `table_images` | dict | Table images as base64 PNGs. Keys: arial, times_new_roman, red, blue, green. Unrendered variants are empty strings. |
| `source` | string | Source dataset and split (e.g. `hitab_train`) |
| `source_id` | string | Original ID from source dataset |
