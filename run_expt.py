import torch
import argparse

from src.utils.seed import set_seed
from src.utils.config import load_config


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description="ViTaB-A Experiment Runner")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--test-mode", action="store_true", help="Quick test with few samples"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config.seed = args.seed

    if args.test_mode:
        config.max_samples = min(config.max_samples or 5, 5)

    set_seed(config.seed)
    print(f"Using seed: {config.seed}")

    device = resolve_device(config.device)
    print(f"Using device: {device}")

    # Dispatch based on task type
    if config.task == "inference":
        from src.inference.runner import run_inference

        results_path = run_inference(config, device)
        print(f"Experiment complete. Results: {results_path}")
    elif config.task == "agentic":
        raise NotImplementedError("Agentic inference not yet implemented")
    elif config.task == "sft":
        raise NotImplementedError("SFT not yet implemented")
    else:
        raise ValueError(f"Unknown task type: {config.task!r}")


if __name__ == "__main__":
    main()
