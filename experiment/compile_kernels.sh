#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

echo "[Compiling] Building intensity_benchmark_kernels.so for RTX 5000 Ada (sm_89)..."
$CUDA_PATH/bin/nvcc -O3 -arch=sm_89 --shared -Xcompiler -fPIC \
    -o "$SCRIPT_DIR/intensity_benchmark_kernels.so" \
    "$SCRIPT_DIR/intensity_benchmark_kernels.cu" \
    -lcudart

echo "[Compiling] Done! Output: $SCRIPT_DIR/intensity_benchmark_kernels.so"
