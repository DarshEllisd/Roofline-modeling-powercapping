#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_DIR=${CUDA_DIR:-/usr/local/cuda}
CUPTI_INC="$CUDA_DIR/targets/x86_64-linux/include"
CUPTI_LIB="$CUDA_DIR/targets/x86_64-linux/lib"
CUDA_INC="$CUDA_DIR/include"
CUDA_LIB="$CUDA_DIR/lib64"
NVML_LIB="/usr/lib/x86_64-linux-gnu"

echo "=========================================================="
echo "Compiling Roofline Plugin (.so) for Workstation Linux"
echo "Target: NVIDIA RTX 5000 Ada (Power Capping Governor)"
echo "CUDA_DIR: $CUDA_DIR"
echo "=========================================================="

CUPTI_SAMPLES_INC="$CUDA_DIR/extras/CUPTI/samples/extensions/include -I$CUDA_DIR/extras/CUPTI/samples/extensions/include/c_util -I$CUDA_DIR/extras/CUPTI/samples/extensions/include/profilerhost_util"

INC="-I. -I$CUDA_INC -I$CUPTI_INC -I$CUPTI_SAMPLES_INC"
LIBS="-L$CUDA_LIB -L$CUPTI_LIB -L$NVML_LIB -lcuda -lcudart -lcupti -lnvperf_host -lnvperf_target -lnvidia-ml"

# Compile with nvcc
nvcc --shared -Xcompiler -fPIC -std=c++17 -O3 \
    -o roofline_plugin.so \
    roofline_plugin.cpp Eval.cpp Metric.cpp \
    $INC $LIBS

echo "=========================================================="
echo "Successfully built roofline_plugin.so!"
echo "=========================================================="
