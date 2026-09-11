#!/usr/bin/env python3
"""
Test Harness for Roofline Plugin on Workstation Linux (NVIDIA RTX 5000 Ada)
Demonstrates:
  1. Automated GPU chip & architecture discovery
  2. CUPTI real-time telemetry (DRAM vs L2 cache traffic, FLOPs, Arithmetic Intensity)
  3. Dynamic power capping governor under <= 8.0% SLA heuristics (No GPU clock locking)
"""

import os
import sys
import time
import ctypes
import torch

def main():
    print("=================================================================")
    print("  Roofline Model Power-Capping Governor Test (Linux Workstation)")
    print("=================================================================")
    
    print("\n[1/4] Initializing PyTorch CUDA Context...")
    torch.cuda.init()
    _ = torch.tensor([1.0], device='cuda') # Force CUDA context initialization
    device_name = torch.cuda.get_device_name(0)
    print(f"  Active GPU: {device_name}")

    # Locate shared library
    base_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(base_dir, 'roofline_plugin.so')
    if not os.path.exists(so_path):
        print(f"\n[ERROR] Shared library not found at: {so_path}")
        print("  Please build it first with: bash compile.sh")
        return 1

    print(f"\n[2/4] Loading plugin: {so_path}")
    plugin = ctypes.CDLL(so_path)

    # Set argtypes/restypes for C API functions
    plugin.start_profiling_auto.restype = None
    plugin.enable_governor.restype = None
    plugin.disable_governor.restype = None
    plugin.stop_profiling.restype = None
    plugin.get_profiler_state.restype = ctypes.c_int
    plugin.get_active_power_cap.restype = ctypes.c_uint32
    plugin.set_power_cap_manual.argtypes = [ctypes.c_uint32]
    plugin.set_power_cap_manual.restype = None

    print("\n[3/4] Starting CUPTI Profiler Background Thread & Enabling Governor...")
    plugin.start_profiling_auto()
    time.sleep(1.0)
    plugin.enable_governor()

    print("\n--- PHASE 1: Running DRAM-BOUND Workload (Large 256MB Tensor Operations) ---")
    dram_tensor = torch.randn(64 * 1024 * 1024, dtype=torch.float32, device='cuda') # 256 MB
    t_end = time.time() + 6.0
    while time.time() < t_end:
        dram_tensor = torch.tanh(dram_tensor) + torch.sigmoid(dram_tensor)
        torch.cuda.synchronize()
        time.sleep(0.05)

    print("\n--- PHASE 2: Running L2-CACHE-RESIDENT Workload (Small 16MB Tensor Operations) ---")
    l2_tensor = torch.randn(4 * 1024 * 1024, dtype=torch.float32, device='cuda') # 16 MB (fits in 64MB L2)
    t_end = time.time() + 6.0
    while time.time() < t_end:
        l2_tensor = torch.tanh(l2_tensor) + torch.sin(l2_tensor)
        torch.cuda.synchronize()
        time.sleep(0.05)

    print("\n--- PHASE 3: Running COMPUTE-BOUND Workload (Dense 4096x4096 FP32 GEMM) ---")
    mat_a = torch.randn(4096, 4096, dtype=torch.float32, device='cuda')
    mat_b = torch.randn(4096, 4096, dtype=torch.float32, device='cuda')
    t_end = time.time() + 6.0
    while time.time() < t_end:
        mat_c = torch.matmul(mat_a, mat_b)
        torch.cuda.synchronize()
        time.sleep(0.05)

    print("\n--- PHASE 4: IDLE State ---")
    time.sleep(3.0)

    print("\n[4/4] Shutting down Profiler & Governor (Restoring Baseline 250W TDP)...")
    plugin.disable_governor()
    plugin.stop_profiling()
    time.sleep(1.0)
    print("\n=== Test Run Completed Successfully! ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
