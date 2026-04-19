import json
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import scan_cache_dir
from playwright.sync_api import sync_playwright
from src.data.table_utils import simplify_table, table_to_markdown, generate_table_images, VARIATIONS

# ── Image generation configuration ──────────────────────────────────
# These can be overridden via CLI arguments.

DEFAULT_IMAGE_MODE = "distribution"
# "all"          - generate all 5 variants for every sample
# "single"       - generate only one variant for all samples
# "distribution" - assign variants by percentage distribution
# "none"         - skip image generation (all variants empty)

DEFAULT_SINGLE_VARIANT = "arial"
# Used when image_mode is "single". Must be a key in VARIATIONS.

DEFAULT_DISTRIBUTION = {
    "arial": 0.2,
    "times_new_roman": 0.2,
    "red": 0.2,
    "blue": 0.2,
    "green": 0.2,
}
# Used when image_mode is "distribution". Values must sum to 1.0.
# Each sample gets exactly one variant based on these probabilities.

README_CONTENT = """\
---
language:
- en
license: apache-2.0
pretty_name: ViTaB-A
task_categories:
- question-answering
tags:
- table-question-answering
configs:
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

hitab = load_dataset("raoanmol/ViTaB-A", "hitab")
fetaqa = load_dataset("raoanmol/ViTaB-A", "fetaqa")
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
| `table_json` | dict | Simplified table with keys: `title` (string), `header` (list of header rows), `rows` (list of data rows) |
| `table_md` | string | Markdown representation of the table with Excel-style row/column labels |
| `table_images` | dict | Table images as base64 PNGs. Keys: arial, times_new_roman, red, blue, green. Unrendered variants are empty strings. |
| `source` | string | Source dataset and split (e.g. `hitab_train`) |
| `source_id` | string | Original ID from source dataset |
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Process FeTaQA dataset into ViTaB-A unified schema.")
    parser.add_argument("--test-mode", action="store_true", help="Process only 20 samples total (~7 per split).")
    parser.add_argument("--output-dir", default="./data", help="Root output directory (default: ./data).")
    parser.add_argument(
        "--image-mode",
        choices=["all", "single", "distribution", "none"],
        default=DEFAULT_IMAGE_MODE,
        help="Image generation mode (default: %(default)s).",
    )
    parser.add_argument(
        "--image-variant",
        default=DEFAULT_SINGLE_VARIANT,
        help="Variant to render in 'single' mode (default: %(default)s).",
    )
    parser.add_argument(
        "--image-distribution",
        default=None,
        help=(
            "Variant distribution for 'distribution' mode. "
            "Format: 'arial=0.2,times_new_roman=0.2,red=0.2,blue=0.2,green=0.2'. "
            "Values must sum to 1.0."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for distribution mode (default: 42).",
    )
    return parser.parse_args()


def parse_distribution(spec: str) -> dict:
    """Parse a distribution spec string like 'arial=0.2,red=0.3,...' into a dict."""
    dist = {}
    for pair in spec.split(","):
        name, weight = pair.strip().split("=")
        name = name.strip()
        weight = float(weight.strip())
        if name not in VARIATIONS:
            raise ValueError(f"Unknown variant: {name!r}. Must be one of {list(VARIATIONS.keys())}")
        dist[name] = weight

    total = sum(dist.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Distribution weights must sum to 1.0, got {total}")

    # Fill missing variants with 0
    for name in VARIATIONS:
        if name not in dist:
            dist[name] = 0.0

    return dist


def pick_variant(distribution: dict) -> str:
    """Select a variant name based on a probability distribution."""
    r = random.random()
    cumulative = 0.0
    for name, weight in distribution.items():
        cumulative += weight
        if r < cumulative:
            return name
    # Fallback to last variant (handles floating point edge case)
    return list(distribution.keys())[-1]


def column_letter(col_idx):
    """Convert 0-indexed column number to Excel-style letter (A, B, ..., Z, AA, AB, ...)."""
    letter = ''
    col_idx += 1
    while col_idx > 0:
        col_idx -= 1
        letter = chr(col_idx % 26 + ord('A')) + letter
        col_idx //= 26
    return letter


def convert_citation(highlighted_cell_ids):
    """Convert [[row, col], ...] to ["=A1", "=B2", ...] format."""
    citations = []
    for cell in highlighted_cell_ids:
        row, col = cell[0], cell[1]
        ref = f"={column_letter(col)}{row + 1}"
        citations.append(ref)
    return citations


def convert_table_to_json(table_array, page_title, section_title):
    """Convert FeTaQA 2D table array to HiTab-style JSON structure."""
    title_parts = []
    if page_title:
        title_parts.append(page_title)
    if section_title:
        title_parts.append(section_title)
    title = " - ".join(title_parts)

    return {
        "texts": table_array,
        "merged_regions": [],
        "top_header_rows_num": 1,
        "left_header_columns_num": 0,
        "title": title,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    fetaqa_dir = output_dir / "fetaqa"
    fetaqa_dir.mkdir(parents=True, exist_ok=True)

    # Resolve image configuration
    image_mode = args.image_mode
    if image_mode == "distribution":
        random.seed(args.seed)
        if args.image_distribution:
            distribution = parse_distribution(args.image_distribution)
        else:
            distribution = DEFAULT_DISTRIBUTION.copy()
    elif image_mode == "single":
        if args.image_variant not in VARIATIONS:
            raise ValueError(f"Unknown variant: {args.image_variant!r}. Must be one of {list(VARIATIONS.keys())}")

    print("Loading FeTaQA dataset from HuggingFace...")
    ds = load_dataset("DongfuJiang/FeTaQA")

    # Launch browser for image generation
    browser = None
    page = None
    pw = None
    if image_mode != "none":
        print("Launching browser for image generation...")
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

    splits = ["train", "validation", "test"]
    test_limit = 7
    counter = 1
    split_counts = {}

    for split_name in splits:
        split_data = ds[split_name]
        if args.test_mode:
            split_data = split_data.select(range(min(test_limit, len(split_data))))

        out_path = fetaqa_dir / f"{split_name}.jsonl"
        print(f"\nProcessing FeTaQA {split_name} split...")

        written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(tqdm(split_data)):
                feta_id = sample.get("feta_id", "")
                try:
                    table_json = convert_table_to_json(
                        sample["table_array"],
                        sample.get("table_page_title", ""),
                        sample.get("table_section_title", ""),
                    )

                    # Warn if any cited cell is out of bounds
                    num_rows = len(sample["table_array"])
                    num_cols = len(sample["table_array"][0]) if num_rows > 0 else 0
                    for cell in sample.get("highlighted_cell_ids", []):
                        r, c = cell[0], cell[1]
                        if r >= num_rows or c >= num_cols:
                            print(
                                f"\n  WARNING: Sample {i} (feta_id={feta_id}) has out-of-bounds "
                                f"cell [{r}, {c}] for table of shape ({num_rows}, {num_cols})"
                            )

                    citation = convert_citation(sample.get("highlighted_cell_ids", []))

                    # Generate markdown
                    table_md = table_to_markdown(table_json)

                    # Determine which image variants to render
                    if image_mode == "all":
                        variants_to_render = list(VARIATIONS.keys())
                    elif image_mode == "single":
                        variants_to_render = [args.image_variant]
                    elif image_mode == "distribution":
                        variants_to_render = [pick_variant(distribution)]
                    else:
                        variants_to_render = []

                    # Generate images
                    if image_mode != "none":
                        table_images = generate_table_images(table_json, page, variants_to_render)
                    else:
                        table_images = {name: "" for name in VARIATIONS}

                    record = {
                        "id": f"vitaba_{counter:06d}_fetaqa",
                        "split": split_name,
                        "question": sample["question"],
                        "answer": sample["answer"],
                        "citation": citation,
                        "table_json": simplify_table(table_json),
                        "table_md": table_md,
                        "table_images": table_images,
                        "source": f"fetaqa_{split_name}",
                        "source_id": str(feta_id),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counter += 1
                    written += 1

                except Exception as e:
                    print(f"\n  WARNING: Failed to process sample {i} (feta_id={feta_id!r}): {e}")
                    continue

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  -> Wrote {written} samples to {out_path} ({size_mb:.2f} MB)")
        split_counts[split_name] = written

    # Close browser
    if browser:
        browser.close()
    if pw:
        pw.stop()

    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(README_CONTENT, encoding="utf-8")
        print(f"\nCreated dataset card at {readme_path}")

    print("\nCleaning up HuggingFace cache for DongfuJiang/FeTaQA...")
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == "DongfuJiang/FeTaQA":
            shutil.rmtree(repo.repo_path)
            print(f"  Cleaned up cache at {repo.repo_path}")
            break
    else:
        print("  No cache entry found for DongfuJiang/FeTaQA.")

    print("\nDone. Samples written per split:")
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count}")
    print(f"  Total: {sum(split_counts.values())}")


if __name__ == "__main__":
    main()
