"""
scripts/check_env.py

Sanity-check the environment right after setup_env.sh, before running
anything real. Confirms:
  - which Python / package versions are actually installed
  - whether CUDA is visible to torch, and basic GPU info
  - that a minimal tensor op + autograd call actually runs on the GPU
    (catches "torch installed but CUDA build mismatch" issues early,
    which torch.cuda.is_available() alone doesn't always catch)

Run:
    python scripts/check_env.py
"""

import platform
import sys

import numpy as np
import torch

print("=" * 60)
print("environment check")
print("=" * 60)
print(f"python:      {sys.version.split()[0]} ({platform.platform()})")
print(f"numpy:       {np.__version__}")
print(f"torch:       {torch.__version__}")
print(f"torch cuda build: {torch.version.cuda}")
print(f"cuda available:   {torch.cuda.is_available()}")
print(f"mps available:    {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"cuda device count: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    print("WARNING: no GPU backend detected — falling back to cpu.")

print("-" * 60)
print(f"running a minimal forward+backward on device='{device}' ...")

try:
    x = torch.rand(100, 1, device=device, requires_grad=True)
    y = torch.sin(x) ** 2
    grad = torch.autograd.grad(y.sum(), x)[0]
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    print(f"  OK: forward+backward succeeded, grad shape={tuple(grad.shape)}, "
          f"grad device={grad.device}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("=" * 60)
print("environment check passed.")
print("=" * 60)
