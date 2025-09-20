#!/bin/sh
# tools/build.sh - wrapper to build WE-FDTD (expects WE-FDTD.cu in repo root or adjust path)
set -e
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Error: nvcc not found in PATH. Install CUDA toolkit or load module."
  exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
  echo "Warning: mpirun not found. You can still build but running multi-process requires OpenMPI."
fi
cd "$(dirname "$0")"
make