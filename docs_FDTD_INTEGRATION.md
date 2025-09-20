# FDTD Integration (WE-FDTD) — bat_sensory

This document describes how to build and integrate the WE-FDTD simulator (CUDA implementation) with `bat_sensory` using the helper scripts in `tools/`.

## Overview

We provide:
- `tools/Makefile` and `tools/build.sh` — compile the WE-FDTD CUDA source (the CUDA source must be present at repo root as `WE-FDTD.cu`, or you can point `WE_FDTD_BIN` to a prebuilt binary).
- `tools/fdtd_runner.py` — Python helper to create an `input.dat`, run the WE-FDTD executable (optionally via `mpirun`) and return parsed waveform data.
- `tools/run_fdtd_batch.py` — batch runner to run many cases (parallel) and collect logs.
- `examples/fdtd_demo` — a minimal example case.

## Prerequisites

- NVIDIA drivers and CUDA toolkit (nvcc) installed on the machine that will run the simulation.
- OpenMPI (`mpirun`) if you intend to run with multiple MPI processes / multi-node.
- A GPU supported by the CUDA arch used at compile time (default `sm_52` in `tools/Makefile`).
- Python 3 for the helper runner scripts.

## Quick build & run (single machine)

1. Place WE-FDTD CUDA source (or compiled binary):
   - Option A (recommended): copy `WE-FDTD.cu` (and any .h) to the repository root.
   - Option B: provide a path to a precompiled binary and set `WE_FDTD_BIN` env var.

2. Build:
   ```sh
   cd tools
   ./build.sh
   ```
   This produces `tools/WE-FDTD`.

3. Run the example:
   ```sh
   # from repo root
   python3 tools/run_fdtd_batch.py --cases examples/fdtd_demo/cases_list.txt --exec ./tools/WE-FDTD --np 1 --parallel 1
   ```

4. Inspect outputs in `outputs/<case_name>` (logs, `wave.csv`, etc).

## Using fdtd_runner from bat_sensory (recommended pattern)

- Use `tools/fdtd_runner.py` to create a per-case directory, copy support files (e.g. `cell.dat`, `SourceWave.csv`), write `input.dat`, run the WE-FDTD executable, and parse the waveform (`wave.csv` or `recv*.csv`).
- Example (from Python):
```py
from tools.fdtd_runner import run_and_get_wave

params = {
  "case_name": "demo1",
  "Nx": 200, "Ny": 200,
  "Nt": 1000,
  "iSource": 0,
  "Src": (100, 50, 1.0, 0, 0, 0),
  "iReceiver": 0,
  "Rcv": (100, 150, 1.0, 0, 0, 0),
  "freq": 0,
}
wave = run_and_get_wave(params, scratch_base="./fdtd_cases", mpirun_np=1)
```

### Notes on coordinate mapping
- `bat_sensory` sampling coordinates must be mapped into the FDTD grid coordinate system:
  - choose `dl` (grid spacing in meters) and an origin mapping (grid (0,0) location).
  - For a sample at physical (x,y) meters: grid_x = int((x - origin_x) / dl)

### GPU binding and concurrency
- For many simultaneous cases: prefer multi-process approach (each process binds to its GPU) rather than trying to call CUDA kernels in the same Python process.
- Use `MPIRUN` or job scheduler (Slurm) to allocate GPUs and run `WE-FDTD` per process.
- If using single GPU and many short simulations, consider batching multiple receiver queries per run to amortize startup cost.

## Troubleshooting
- If you see NaNs or divergence: check CFL (`cfl`, `dl`, `c0`) and reduce dt or adjust `cfl`.
- If `nvcc` fails: adjust `ARCH` in `tools/Makefile` or remove `-arch` to let nvcc choose.
- If `mpirun` missing: install OpenMPI or run single-process tests.

## Next steps for tighter integration
- For high-throughput/low-latency use, consider:
  - Creating a service that runs WE-FDTD once and serves multiple small simulation requests via in-memory IPC or socket calls.
  - Building a shared library (.so) wrapper for the core step (advanced, requires refactor of CUDA/MPI patterns).