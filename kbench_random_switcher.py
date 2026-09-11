#!/usr/bin/env python3
"""
Stress-Test Dynamic State Transitions on Real-World PyTorch KBENCH Workloads
Platform: NVIDIA RTX 5000 Ada Generation (Linux Workstation)
Uses Roofline Plugin Power Capping Governor under <= 8.0% Latency SLA
(No GPU Clock Locking - Pure Power Caps [100W - 250W])
"""

import os
import sys

# Auto-detect and re-exec into the CUDA-enabled kernel-bench environment if needed
KERNEL_BENCH_PYTHON = "/home/antpc/anaconda3/envs/kernel-bench/bin/python"
if os.path.exists(KERNEL_BENCH_PYTHON) and sys.executable != KERNEL_BENCH_PYTHON:
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"[INFO] Current Python ({sys.executable}) lacks CUDA support. Auto-switching to kernel-bench environment...")
            os.execv(KERNEL_BENCH_PYTHON, [KERNEL_BENCH_PYTHON] + sys.argv)
    except Exception:
        print(f"[INFO] Missing CUDA dependencies in {sys.executable}. Auto-switching to kernel-bench environment...")
        os.execv(KERNEL_BENCH_PYTHON, [KERNEL_BENCH_PYTHON] + sys.argv)

import time
import glob
import random
import importlib.util
import ctypes
import torch
import pynvml

def load_workload_module(file_path):
    spec = importlib.util.spec_from_file_location("workload_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    print("================================================================================")
    print("  KBENCH DYNAMIC WORKLOAD SWITCHER: ROOFLINE POWER CAPPING GOVERNOR")
    print("  Platform: NVIDIA RTX 5000 Ada Generation (Workstation)")
    print("  Governor Policy: Dynamic Power Caps [100W, 250W] | SLA <= 8.0%")
    print("================================================================================")

    print("\n[1/4] Initializing PyTorch CUDA context...")
    torch.cuda.init()
    _ = torch.tensor([1.0], device='cuda')
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 1. Load Roofline Plugin Shared Library (.so on Linux, .dll on Windows)
    plugin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roofline_plugin')
    
    if sys.platform == "win32":
        lib_name = 'roofline_plugin.dll'
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\extras\CUPTI\lib64")
            os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\bin")
            os.add_dll_directory(plugin_dir)
    else:
        lib_name = 'roofline_plugin.so'

    lib_path = os.path.join(plugin_dir, lib_name)
    print(f"\n[2/4] Loading Roofline Plugin from: {lib_path}")
    
    plugin = None
    if os.path.exists(lib_path):
        try:
            plugin = ctypes.CDLL(lib_path)
            plugin.start_profiling_auto.restype = None
            plugin.enable_governor.restype = None
            plugin.disable_governor.restype = None
            plugin.stop_profiling.restype = None
            plugin.get_profiler_state.restype = ctypes.c_int
            plugin.get_active_power_cap.restype = ctypes.c_uint32
            print("  Successfully loaded plugin.")
        except Exception as e:
            print(f"  Note: Could not load shared library ({e}). Running in monitor mode.")
    else:
        print(f"  Note: Library {lib_path} not found. Running in monitor mode.")

    # 2. Start Profiler & Enable Power Capping Governor
    if plugin:
        print("\n>>> STARTING TRUE CUPTI ROOFLINE PROFILER (AUTO-DISCOVERY) <<<")
        plugin.start_profiling_auto()
        time.sleep(1.0)

        print(">>> ENABLING DYNAMIC GOLDEN ZONE POWER CAPPING GOVERNOR (SLA <= 8.0%) <<<")
        plugin.enable_governor()
        time.sleep(1.0)

    # 3. Discover workloads
    base_dir = os.path.dirname(os.path.abspath(__file__))
    compute_dir = os.path.join(base_dir, 'workloads_KBENCH_EVAL', 'compute')
    memory_dir  = os.path.join(base_dir, 'workloads_KBENCH_EVAL', 'memory')

    compute_files = sorted(glob.glob(os.path.join(compute_dir, "*.py")))
    memory_files  = sorted(glob.glob(os.path.join(memory_dir, "*.py")))

    print(f"\n[3/4] Discovered {len(compute_files)} Compute workloads and {len(memory_files)} Memory workloads.")

    # 4. Alternate between Compute and Memory workloads
    num_rounds = 10
    print(f"\n[4/4] Starting {num_rounds} Alternating Workload Rounds...")

    state_names = {0: "IDLE", 1: "DRAM-BOUND", 2: "L2-BOUND", 3: "COMPUTE-BOUND"}

    for round_idx in range(1, num_rounds + 1):
        category = "compute" if round_idx % 2 == 1 else "memory"
        candidates = compute_files if category == "compute" else memory_files
        chosen_file = random.choice(candidates)
        filename = os.path.basename(chosen_file)

        print(f"\n================================================================================")
        print(f" ROUND {round_idx}/{num_rounds} | EXPECTED REGIME: [{category.upper()}-BOUND]")
        print(f" RUNNING WORKLOAD: {filename}")
        print(f"================================================================================")

        try:
            mod = load_workload_module(chosen_file)

            if hasattr(mod, 'batch_size') and mod.batch_size > 1024:
                mod.batch_size = 1024
            if hasattr(mod, 'dim') and mod.dim > 131072:
                mod.dim = 131072

            init_inputs = mod.get_init_inputs() if hasattr(mod, 'get_init_inputs') else []
            model = mod.Model(*init_inputs).cuda()
            model.eval()

            raw_inputs = mod.get_inputs() if hasattr(mod, 'get_inputs') else [torch.randn(512, 1024)]
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in raw_inputs]

            # Run workload loop for 5.0 seconds
            t_start = time.time()
            iters = 0
            powers = []
            clocks = []

            with torch.no_grad():
                while time.time() - t_start < 5.0:
                    model(*inputs)
                    torch.cuda.synchronize()
                    iters += 1
                    p = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                    powers.append(p)
                    clk = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
                    clocks.append(clk)
                    time.sleep(0.01)

            avg_pwr = sum(powers) / len(powers) if powers else 0.0
            avg_clk = sum(clocks) / len(clocks) if clocks else 0.0
            
            gov_state = plugin.get_profiler_state() if plugin else 0
            active_cap = plugin.get_active_power_cap() if plugin else 250

            print(f"  Summary: Executed {iters} iters in 5.0s")
            print(f"  Average Power: {avg_pwr:.1f} W | Avg Core Clock: {avg_clk:.0f} MHz (Native GPU Boost)")
            print(f"  Governor State: [{state_names.get(gov_state, 'UNKNOWN')}] | Active Power Cap: {active_cap} W")

            # Cleanup
            del model, inputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Error executing {filename}: {e}")

        time.sleep(0.5)

    print("\n================================================================================")
    if plugin:
        print("Stopping Profiler & Restoring Baseline Power Cap (250W)...")
        plugin.disable_governor()
        plugin.stop_profiling()
        time.sleep(1.0)

    try:
        cur_power_cap = pynvml.nvmlDeviceGetPowerManagementLimit(nvml_handle) / 1000.0
        print(f"Final Power Cap: {cur_power_cap:.0f} W (Baseline TDP Restored)")
    except Exception:
        pass

    print("Dynamic KBENCH switcher benchmark complete.")

if __name__ == '__main__':
    main()
