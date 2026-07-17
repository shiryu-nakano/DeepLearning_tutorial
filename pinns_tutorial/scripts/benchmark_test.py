"""
Benchmark PINN training speed across devices.

Run this on your MacBook (will test cpu, and mps if you're on Apple
Silicon) and separately on a CUDA machine (will test cpu and cuda).
It reports:
  - seconds/epoch and epochs/sec for each available device
  - an estimated wall-clock time for the full run_sweep.py sweep
    (N_RUNS runs x N_EPOCHS_SWEEP epochs each), so you can decide
    where to actually run the sweep before committing to it.

Usage:
    python benchmark_speed.py
    python benchmark_speed.py --devices cpu mps        # force which devices to test
    python benchmark_speed.py --epochs 300 --warmup 30 # change timing budget

Notes:
  - Timed epochs use the exact same forward/backward/step logic as
    pinn_heat_lib.train_pinn (same model size, same point counts), just
    without the periodic true_L2 eval or disk I/O, so the timing isolates
    the actual training-step cost.
  - GPU/MPS timing requires an explicit synchronize() call before reading
    the clock, otherwise you're only timing kernel *launch*, not
    completion (async execution makes naive timing wrong).
  - First few iterations on any device (especially CUDA/MPS) pay a
    one-time warmup cost (kernel compilation/caching), so those are
    excluded from the timed average.
"""

import argparse
import time

import numpy as np
import torch

from pinn_heat_lib import PINN

# these should match what run_sweep.py / pinn_heat_lib.py actually use,
# so the benchmark reflects the real workload
ALPHA, L, T = 1.0, 1.0, 0.2
WIDTH, DEPTH = 32, 4
N_PDE, N_IC, N_BC = 2000, 200, 200
LR = 1e-3

# for the runtime estimate at the end
N_RUNS_SWEEP = 25       # seeds x weights in run_sweep.py
N_EPOCHS_SWEEP = 3000


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    # cpu: no-op, it's already synchronous


def run_one_epoch(model, optimizer, x_ic, t_ic, u_ic_target, x_bc0, x_bc1, t_bc, device):
    optimizer.zero_grad()

    u_ic_pred = model(x_ic, t_ic)
    L_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)

    u_bc0 = model(x_bc0, t_bc)
    u_bc1 = model(x_bc1, t_bc)
    L_bc = torch.mean(u_bc0**2) + torch.mean(u_bc1**2)

    x_f = torch.rand(N_PDE, 1, device=device, requires_grad=True) * L
    t_f = torch.rand(N_PDE, 1, device=device, requires_grad=True) * T
    u_f = model(x_f, t_f)
    u_t = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    residual = u_t - ALPHA * u_xx
    L_pde = torch.mean(residual**2)

    loss = 10.0 * L_ic + 10.0 * L_bc + 1.0 * L_pde
    loss.backward()
    optimizer.step()


def benchmark_device(device, n_epochs, n_warmup):
    torch.manual_seed(0)
    np.random.seed(0)

    model = PINN(width=WIDTH, depth=DEPTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x_ic = torch.rand(N_IC, 1, device=device) * L
    t_ic = torch.zeros(N_IC, 1, device=device)
    u_ic_target = torch.sin(np.pi * x_ic)
    t_bc = torch.rand(N_BC, 1, device=device) * T
    x_bc0 = torch.zeros(N_BC, 1, device=device)
    x_bc1 = torch.ones(N_BC, 1, device=device)

    # warmup: not timed, absorbs kernel compilation / caching cost
    for _ in range(n_warmup):
        run_one_epoch(model, optimizer, x_ic, t_ic, u_ic_target, x_bc0, x_bc1, t_bc, device)
    sync(device)

    start = time.perf_counter()
    for _ in range(n_epochs):
        run_one_epoch(model, optimizer, x_ic, t_ic, u_ic_target, x_bc0, x_bc1, t_bc, device)
    sync(device)
    elapsed = time.perf_counter() - start

    sec_per_epoch = elapsed / n_epochs
    epochs_per_sec = n_epochs / elapsed
    return sec_per_epoch, epochs_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=None,
                         help="e.g. --devices cpu mps  (default: auto-detect all available)")
    parser.add_argument("--epochs", type=int, default=300, help="timed epochs per device")
    parser.add_argument("--warmup", type=int, default=30, help="untimed warmup epochs per device")
    args = parser.parse_args()

    devices = args.devices if args.devices else available_devices()
    print(f"testing devices: {devices}")
    print(f"(timed epochs={args.epochs}, warmup={args.warmup}, "
          f"model=width{WIDTH}x{DEPTH}, n_pde={N_PDE})\n")

    results = {}
    for device in devices:
        try:
            sec_per_epoch, epochs_per_sec = benchmark_device(device, args.epochs, args.warmup)
        except Exception as e:
            print(f"  {device}: FAILED ({e})")
            continue
        results[device] = sec_per_epoch
        est_sweep_hours = (sec_per_epoch * N_EPOCHS_SWEEP * N_RUNS_SWEEP) / 3600
        print(f"  {device:>6}: {sec_per_epoch*1000:7.2f} ms/epoch  "
              f"({epochs_per_sec:7.1f} epoch/s)   "
              f"est. full sweep ({N_RUNS_SWEEP} runs x {N_EPOCHS_SWEEP} epochs): "
              f"{est_sweep_hours:.2f} h")

    if len(results) >= 2 and "cpu" in results:
        print("\nspeedup vs cpu:")
        for device, spe in results.items():
            if device == "cpu":
                continue
            print(f"  {device} is {results['cpu'] / spe:.1f}x faster than cpu")


if __name__ == "__main__":
    main()