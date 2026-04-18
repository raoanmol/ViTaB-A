import torch
import argparse

from src.utils.seed import set_seed
from src.utils.config import load_config

def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type = str)
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--test-mode', action = 'store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)
    print(f'Using seed: {config.seed}')

    device = resolve_device(config.device)
    print(f'Using device: {config.device}')

if __name__ == '__main__':
    main()