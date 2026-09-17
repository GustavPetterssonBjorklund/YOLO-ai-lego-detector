#!/usr/bin/env python3
"""Report the PyTorch accelerator available to Ultralytics."""

import torch


def main() -> None:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
