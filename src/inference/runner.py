import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.dataset import load_dataset
from src.models.factory import create_model
from src.prompts.builder import build_prompt
from src.utils.config import ExperimentConfig


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def run_inference(config: ExperimentConfig, device: torch.device) -> Path:
    """Run the full inference pipeline.

    Args:
        config: Experiment configuration
        device: Resolved torch device

    Returns:
        Path to the predictions JSONL file.
    """
    dtype = DTYPE_MAP[config.dtype]

    # 1. Load model
    print(f"Loading model: {config.model}")
    model_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
    }
    if config.model_name_override:
        model_kwargs["model_name"] = config.model_name_override

    # Import model modules to trigger registration
    import src.models

    model = create_model(
        name=config.model,
        device=device,
        dtype=dtype,
        **model_kwargs,
    )
    print(f"Model loaded: {model.short_name}")

    # 2. Load data
    print(f"Loading dataset: {config.dataset}/{config.split}")
    samples = load_dataset(
        config.data_dir, config.dataset, config.split, config.max_samples
    )
    print(f"Loaded {len(samples)} samples")

    # 3. Prepare output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{model.short_name}_{config.dataset}_{config.table_repr}_{timestamp}"
    run_dir = Path(config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save run metadata
    meta = {
        "experiment_name": config.experiment_name,
        "model": config.model,
        "model_name": model.model_name,
        "dataset": config.dataset,
        "split": config.split,
        "table_repr": config.table_repr,
        "prompt_strategy": config.prompt_strategy,
        "num_samples": len(samples),
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "dtype": config.dtype,
        "device": str(device),
        "seed": config.seed,
        "start_time": timestamp,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # 5. Run inference loop
    results_path = run_dir / "predictions.jsonl"
    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0

    with open(results_path, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"Inference ({model.short_name})"):
            try:
                prompt_text, image = build_prompt(
                    sample, config.table_repr, config.prompt_strategy
                )
                output = model.generate(prompt_text, image)

                record = {
                    "sample_id": sample.id,
                    "question": sample.question,
                    "answer": sample.answer,
                    "ground_truth": sample.citation,
                    "model_output": output.raw_text,
                    "predicted_citations": output.parsed_citations,
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                }
                total_input_tokens += output.input_tokens
                total_output_tokens += output.output_tokens

            except Exception as e:
                record = {
                    "sample_id": sample.id,
                    "question": sample.question,
                    "answer": sample.answer,
                    "ground_truth": sample.citation,
                    "model_output": None,
                    "predicted_citations": [],
                    "error": str(e),
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                errors += 1
                print(f"\nError on {sample.id}: {e}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # Crash resilience: flush after each sample

    # 6. Save summary
    end_time = time.strftime("%Y%m%d_%H%M%S")
    summary = {
        **meta,
        "end_time": end_time,
        "total_samples": len(samples),
        "errors": errors,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # 7. Cleanup
    model.unload()
    print(f"\nResults saved to {results_path}")
    print(f"Processed {len(samples)} samples ({errors} errors)")
    return results_path
