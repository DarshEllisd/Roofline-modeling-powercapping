import os
import time
import ctypes
import torch
import pynvml

def main():
    print("================================================================================")
    print(" VERIFICATION: TRUE CUPTI ROOFLINE + DYNAMIC GOLDEN ZONE GOVERNOR")
    print("================================================================================")

    # 1. Initialize CUDA
    torch.cuda.init()
    _ = torch.tensor([1.0], device='cuda')
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    # 2. Setup NVML to monitor live power and clocks from Python
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 3. Load Roofline Plugin DLL
    dll_dir = os.path.join(os.path.dirname(__file__), 'roofline_plugin')
    os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\extras\CUPTI\lib64")
    os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\bin")
    os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\lib\x64")
    os.add_dll_directory(dll_dir)

    dll_path = os.path.join(dll_dir, 'roofline_plugin.dll')
    plugin = ctypes.CDLL(dll_path)

    # 4. Start Profiler & Enable Golden Zone Governor (945 MHz Memory, 1950 MHz Compute)
    plugin.start_profiling_auto()
    time.sleep(1.0)

    print("\n>>> ENABLING GOLDEN ZONE GOVERNOR (Memory: 945 MHz | Compute: 1950 MHz) <<<")
    plugin.enable_governor(945, 1950)
    time.sleep(1.0)

    # -------------------------------------------------------------------------
    # PHASE 1: Memory-Bound Workload (Element-wise Tanh on 16M floats)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print(" PHASE 1: RUNNING MEMORY-BOUND WORKLOAD (Tanh on 16M floats)")
    print(" Expected: Classify as MEMORY-BOUND -> Governor locks to 945 MHz")
    print("================================================================================")
    
    t_mem = torch.randn(16 * 1024 * 1024, device='cuda')
    t0 = time.time()
    mem_powers = []
    with torch.no_grad():
        while time.time() - t0 < 6.5:
            _ = torch.tanh(t_mem)
            torch.cuda.synchronize()
            p = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
            mem_powers.append(p)
            time.sleep(0.02)

    avg_mem_pwr = sum(mem_powers) / len(mem_powers) if mem_powers else 0.0
    active_clock_1 = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
    print(f"\n[PHASE 1 SUMMARY] Actual Core Clock: {active_clock_1} MHz | Average Power: {avg_mem_pwr:.1f} W")

    time.sleep(1.0)

    # -------------------------------------------------------------------------
    # PHASE 2: Compute-Bound Workload (Heavy 4096 x 4096 FP32 GEMM)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print(" PHASE 2: RUNNING COMPUTE-BOUND WORKLOAD (4096 x 4096 GEMM)")
    print(" Expected: Classify as COMPUTE-BOUND -> Governor locks to 1950 MHz")
    print("================================================================================")

    mat_a = torch.randn(4096, 4096, device='cuda')
    mat_b = torch.randn(4096, 4096, device='cuda')
    t0 = time.time()
    comp_powers = []
    with torch.no_grad():
        while time.time() - t0 < 6.5:
            _ = torch.matmul(mat_a, mat_b)
            torch.cuda.synchronize()
            p = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
            comp_powers.append(p)
            time.sleep(0.01)

    avg_comp_pwr = sum(comp_powers) / len(comp_powers) if comp_powers else 0.0
    active_clock_2 = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
    print(f"\n[PHASE 2 SUMMARY] Actual Core Clock: {active_clock_2} MHz | Average Power: {avg_comp_pwr:.1f} W")

    time.sleep(1.0)

    # -------------------------------------------------------------------------
    # CLEANUP & SHUTDOWN
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print(" Stopping Profiler & Releasing Clocks...")
    plugin.stop_profiling()
    time.sleep(1.0)

    final_clock = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
    print(f"Final Released Core Clock: {final_clock} MHz (Driver Default)")
    print("================================================================================")
    print(" TEST COMPLETE.")

if __name__ == '__main__':
    main()
