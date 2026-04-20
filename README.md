# ViTAB

Unified benchmark framework for VisualCite table-cell attribution across multiple vision-language models.

ViTAB evaluates how well models identify the supporting table cell(s) for a question-answer pair, with support for:
- Multiple model families: Gemma, InternVL 3.5, Qwen3-VL, Molmo2
- Multiple table representations: JSON, Markdown, and styled images
- Multiple prompting strategies: zero-shot, few-shot, chain-of-thought
- Confidence analysis: internal confidence, verbalized certainty, alignment metrics
- Uncertainty quantification: split conformal prediction (LAC + APS)

## Project Structure

This repository is consolidated into a single source directory:

```text
ViTAB-A/
├── README.md
└── src/
		├── benchmark_runner.py
		├── confidence_benchmark_runner.py
		├── model_handler.py
		├── data_loader.py
		├── prompt_builder.py
		├── metrics.py
		├── uncertainty_quantification.py
		├── test_benchmark.py
		└── requirements.txt
```

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for model inference
- VisualCite dataset file in JSONL format

Install dependencies:

```bash
cd src
pip install -r requirements.txt
```

## Dataset Setup

By default, the runner expects:

```text
../visualcite.jsonl
```

relative to `src/`.

So with the default configuration, place your dataset at:

```text
ViTAB-A/visualcite.jsonl
```

You can override this with `--jsonl-path`.

## Quick Start

Run a small smoke test from `src/`:

```bash
cd src
python benchmark_runner.py \
	--models Qwen/Qwen3-VL-2B-Instruct \
	--representations markdown \
	--strategies zero_shot \
	--max-samples 5
```

## Running `benchmark_runner.py` Properly

`benchmark_runner.py` uses relative defaults (especially for `--jsonl-path`), so run it in one of these two ways:

### Option A (recommended): run from `src/`

```bash
cd src
python benchmark_runner.py \
	--jsonl-path ../visualcite.jsonl \
	--models Qwen/Qwen3-VL-2B-Instruct \
	--representations markdown \
	--strategies zero_shot \
	--max-samples 5
```

### Option B: run from repository root

```bash
cd ..  # if you are currently inside src
python src/benchmark_runner.py \
	--jsonl-path ./visualcite.jsonl \
	--models Qwen/Qwen3-VL-2B-Instruct \
	--representations markdown \
	--strategies zero_shot \
	--max-samples 5
```

Important:
- If you run from root, do **not** rely on the default `--jsonl-path`.
- Always set `--jsonl-path` explicitly when your working directory is not `src/`.

Run a larger benchmark:

```bash
cd src
python benchmark_runner.py \
	--models Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-4B-Instruct \
	--representations json markdown image_arial \
	--strategies zero_shot few_shot chain_of_thought \
	--max-samples 200 \
	--split dev
```

## Core CLI Options

Common options:
- `--models`: one or more HF model IDs
- `--representations`: `json`, `markdown`, `image_arial`, `image_times_new_roman`, `image_red`, `image_blue`, `image_green`
- `--strategies`: `zero_shot`, `few_shot`, `chain_of_thought`
- `--jsonl-path`: path to dataset JSONL
- `--max-samples`: maximum samples to process
- `--split`: `train`, `validation`, `dev`, `test`
- `--single-cell-only`: filter to samples with exactly one ground-truth cell

Runtime and resume:
- `--output-dir`: output directory (default `benchmark_results`)
- `--checkpoint-dir`: checkpoint directory (default `checkpoints`)
- `--no-resume`: disable checkpoint resume
- `--clear-checkpoints`: remove previous checkpoints before run

Model/runtime settings:
- `--device`: `cuda` or `cpu`
- `--dtype`: `float16`, `bfloat16`, `float32`
- `--no-flash-attention`

Confidence/UQ options:
- `--extract-internal-confidence`
- `--extract-verbalized-certainty`
- `--cqp-template`, `--cqp-max-tokens`, `--cqp-temperature`
- `--enable-conformal-uq`
- `--conformal-calibration-ratio`, `--conformal-alpha`, `--conformal-seed`

## Testing

Run lightweight validation tests:

```bash
cd src
python test_benchmark.py --test all
```

Or specific test groups:

```bash
python test_benchmark.py --test metrics
python test_benchmark.py --test prompts
python test_benchmark.py --test parsing
python test_benchmark.py --test checkpoint
```

## Outputs

By default, outputs are written under `src/benchmark_results/`:
- instance logs (per sample)
- verbalized certainty logs
- aggregated summaries (JSON)
- CSV exports
- markdown report
- uncertainty summaries (if enabled)

Checkpoints are stored under `src/checkpoints/`.

## Notes

- Running from `src/` is recommended because default paths are defined relative to that directory.
- If few-shot validation examples are unavailable at the dataset path, the prompt builder falls back to placeholder examples.
- Large models may require significant GPU memory and can offload to CPU/disk automatically.

# Minor-er Issues:
1. JSON prompt needs tweaking
2. `src/config.py:58-63` defaul list only includes Qwen3-VL. If we run with default, we'll only benchmark Qwen3-VL



# TODO:
1. Implement Brier Score calculator
2. Implement Family Scaling Score (FSS)
3. Update CLI defaults for `--max-samples` (default 500, actual 200) and `--single-cell-only` (default False, actual True)


# Paper Issues:
## Issue [1] Internal confidence not normalized across all cells
- **Files:** `src/confidence_extractor.py`

- **Description:** It's supposed to be $P_{IC}(c) = \frac{P(c)}{\sum_{c' \in C} P(c')}$ (normalize the predicted cell's probability against the sum across ALL cells.) The code currently just calculates the geometric mean of token probabilities for each predicted cell but doesn't normalize against other cells. `compute_aggrergate_confidence()` takes mean/max/min of per-cell values. The `compute_all_cell_probabilities()` function assigns 0.0 to all non-predicted cells (we can only compute the probabilities only for the path the model actually took), making normalization impossible anyway. Ultimately, the confidence scores can't be comparable across different table sizes.

- **Fix:** Update formula in the paper.

---

# Developer Guide

This section covers the current implementation: how to set up the environment, run experiments, understand the codebase, and extend it.

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd ViTaB-A

# Create and activate a virtual environment
python -m venv vitab_venv
source vitab_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The dataset is already preprocessed and stored locally under `data/`. No download step is needed — the JSONL files are read directly at runtime.

## Project Structure

```text
ViTaB-A/
├── run_expt.py              # Single entry point for all experiments
├── requirements.txt
├── configs/                 # YAML experiment configs (one per run)
│   ├── qwen3vl_hitab_markdown.yaml
│   ├── gemma4_hitab_image_arial.yaml
│   ├── internvl3_fetaqa_json.yaml
│   └── molmo2_hitab_markdown.yaml
├── data/
│   ├── hitab/               # HiTab QA dataset (10,672 samples)
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── fetaqa/              # FeTaQA dataset (10,330 samples)
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── results/                 # Created at runtime, one subdirectory per run
├── tests/                   # pytest test suite
└── src/
    ├── data/
    │   ├── dataset.py       # JSONL loader → ViTaBSample dataclass
    │   └── table_utils.py   # Table format conversion utilities
    ├── models/
    │   ├── base.py          # BaseVLM abstract class + ModelOutput dataclass
    │   ├── factory.py       # Model registry + create_model()
    │   ├── qwen3vl.py       # Qwen3-VL-4B wrapper
    │   ├── gemma4.py        # Gemma 3 4B wrapper
    │   ├── internvl3.py     # InternVL3-4B wrapper
    │   └── molmo2.py        # Molmo-7B-D wrapper
    ├── prompts/
    │   └── builder.py       # Prompt construction for all table representations
    ├── inference/
    │   └── runner.py        # Inference orchestration loop
    └── utils/
        ├── config.py        # ExperimentConfig dataclass + load_config()
        ├── seed.py          # set_seed()
        └── parsing.py       # parse_citations() — extracts =ColRow refs from model output
```

## Running an Experiment

All experiments are driven by YAML config files and launched through `run_expt.py`:

```bash
python run_expt.py configs/qwen3vl_hitab_markdown.yaml
```

Use `--test-mode` to cap at 5 samples for a quick sanity check before committing to a full run:

```bash
python run_expt.py configs/qwen3vl_hitab_markdown.yaml --test-mode
```

Override the random seed on the command line:

```bash
python run_expt.py configs/qwen3vl_hitab_markdown.yaml --seed 123
```

Results are written to `results/<model>_<dataset>_<repr>_<timestamp>/`:
- `predictions.jsonl` — one record per sample with model output and parsed citations
- `run_meta.json` — config snapshot written at the start of the run
- `run_summary.json` — token counts and error counts written at the end

The inference loop writes and flushes after every sample, so a crashed run preserves all completed predictions.

## Config Reference

Every YAML field has a default, so a config only needs to specify what differs. Full schema:

| Field | Default | Options |
|-------|---------|---------|
| `experiment_name` | `"default"` | any string |
| `model` | `"qwen3vl"` | `qwen3vl`, `gemma4`, `internvl3`, `molmo2` |
| `model_name_override` | `null` | any HuggingFace model ID (overrides the default HF ID for that model family) |
| `dataset` | `"hitab"` | `hitab`, `fetaqa` |
| `split` | `"test"` | `train`, `validation`, `test` |
| `data_dir` | `"data"` | path to the data root directory |
| `max_samples` | `null` (all) | any integer |
| `table_repr` | `"markdown"` | `json`, `markdown`, `image_arial`, `image_times_new_roman`, `image_red`, `image_blue`, `image_green` |
| `prompt_strategy` | `"zero_shot"` | `zero_shot` (others planned) |
| `max_new_tokens` | `512` | integer |
| `temperature` | `0.0` | float (`0.0` = greedy) |
| `device` | `"auto"` | `auto`, `cuda`, `mps`, `cpu` |
| `dtype` | `"bfloat16"` | `float16`, `bfloat16`, `float32` |
| `seed` | `42` | integer |
| `output_dir` | `"results"` | path |
| `task` | `"inference"` | `inference` (agentic and sft planned) |

Example minimal config — only override what you need:

```yaml
experiment_name: my_run
model: internvl3
dataset: fetaqa
split: validation
table_repr: json
max_samples: 100
```

## Dataset Schema

Each JSONL record represents one table QA sample. After loading via `src/data/dataset.py`, each sample is a `ViTaBSample` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique ID, e.g. `vitaba_000001_hitab` |
| `question` | `str` | Natural language question about the table |
| `answer` | `list` or `str` | Ground truth answer (list for HiTab, string for FeTaQA) |
| `citation` | `list[str]` | Ground truth cell references, e.g. `["=E7"]` |
| `table_json` | `dict` | Structured table with `title`, `header`, `rows` |
| `table_md` | `str` | Markdown table with Excel-style row/column labels |
| `table_images` | `dict[str, str]` | Base64 PNG images keyed by variant (`arial`, `times_new_roman`, `red`, `blue`, `green`). Empty string if not rendered. |
| `source` | `str` | Source dataset + split, e.g. `hitab_test` |
| `source_id` | `str` | Original ID in the source dataset |

The cell reference system used in `citation` and `table_md` matches: column letters (A, B, …, Z, AA, …) and row numbers starting from 1 for the header label row. So `=C4` means column C, row 4 in the rendered markdown table.

## How the Inference Pipeline Works

```
run_expt.py
  └─ load_config(yaml)          # ExperimentConfig dataclass
  └─ set_seed / resolve_device
  └─ run_inference(config, device)
       ├─ create_model(name)    # Loads weights from HuggingFace
       ├─ load_dataset(...)     # Reads JSONL line by line
       └─ for each sample:
            ├─ build_prompt(sample, table_repr)
            │    └─ returns (prompt_text, Optional[PIL.Image])
            ├─ model.generate(prompt_text, image)
            │    └─ returns ModelOutput(raw_text, parsed_citations, ...)
            └─ write record to predictions.jsonl
```

The prompt instructs the model to identify which table cell(s) support a given question-answer pair and return them as Excel-style references (e.g. `=E7`). The `parse_citations` utility then extracts any `=ColRow` patterns from the raw model output.

## Model Architecture

All four model families share the same interface via `BaseVLM`:

```
BaseVLM (abstract)
  ├── load()            — download + load weights + processor onto device
  ├── build_messages()  — format (prompt_text, image) into model-specific chat structure
  ├── generate()        — full inference: tokenize → generate → decode → parse citations
  ├── unload()          — free GPU memory (concrete default, can be overridden)
  └── short_name        — identifier used in result directory names
```

Each model has a different internal loading and inference pattern:

| Model | Class | HF ID | Key difference |
|-------|-------|--------|----------------|
| `qwen3vl` | `Qwen3VLModel` | `Qwen/Qwen3-VL-4B-Instruct` | Uses `qwen_vl_utils.process_vision_info` for image tensors |
| `gemma4` | `Gemma4Model` | `google/gemma-3-4b-it` | Processor's `apply_chat_template` handles images natively |
| `internvl3` | `InternVL3Model` | `OpenGVLab/InternVL3-4B` | Uses `trust_remote_code`, `<image>` tag placeholder, `model.chat()` API |
| `molmo2` | `Molmo2Model` | `allenai/Molmo-7B-D-0924` | Uses `trust_remote_code`, `processor.process()`, `model.generate_from_batch()` |

New models are registered with one line:

```python
register_model("mymodel", MyModelClass, "org/model-id-on-hf")
```

## Running Tests

The test suite uses pytest. Tests that require a GPU are marked `@pytest.mark.slow` and skipped automatically on CPU-only machines.

```bash
# Fast tests only (no GPU required, ~12s)
python -m pytest tests/ -k "not slow" -v

# Full suite including GPU inference tests
python -m pytest tests/ -v --timeout=600
```

Test files map directly to modules:

| Test file | Module under test |
|-----------|-------------------|
| `tests/test_utils.py` | `src/utils/{seed,config,parsing}.py` |
| `tests/test_dataset.py` | `src/data/dataset.py` |
| `tests/test_prompt_builder.py` | `src/prompts/builder.py` |
| `tests/test_model_base.py` | `src/models/{base,factory}.py` |
| `tests/test_qwen3vl.py` | `src/models/qwen3vl.py` |
| `tests/test_gemma4.py` | `src/models/gemma4.py` |
| `tests/test_internvl3.py` | `src/models/internvl3.py` |
| `tests/test_molmo2.py` | `src/models/molmo2.py` |
| `tests/test_runner.py` | `src/inference/runner.py`, `run_expt.py` |

## Extending the Framework

**Adding a new model:**
1. Create `src/models/mymodel.py` implementing `BaseVLM` (`load`, `build_messages`, `generate`, `short_name`)
2. Call `register_model("mymodel", MyModelClass, "hf/model-id")` at the bottom of that file
3. Add `from src.models.mymodel import MyModelClass` to `src/models/__init__.py`
4. Add a config YAML in `configs/`

**Adding a new table representation:**
- Add a branch in `src/prompts/builder.py::get_table_content()` that returns `(text, None)` or `("", image)`

**Adding a new prompting strategy:**
- Add a branch in `src/prompts/builder.py::build_prompt()` keyed on the `strategy` argument

**Adding a new task type (e.g. agentic inference):**
1. Create `src/inference/agentic_runner.py` with a `run_agentic(config, device)` function
2. Add an `elif config.task == "agentic"` branch in `run_expt.py`
3. Add the new task's config fields to `ExperimentConfig` in `src/utils/config.py`

