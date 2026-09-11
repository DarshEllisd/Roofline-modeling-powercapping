import os
import time
import glob
import random
import importlib.util
import ctypes
import torch

def load_workload_module(file_path):
    spec = importlib.util.spec_from_file_location("workload_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    print("Initializing PyTorch CUDA context...")
    torch.cuda.init()
    _ = torch.tensor([1.0], device='cuda')
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # 1. Setup CUPTI search path and load Roofline Plugin DLL
    dll_dir = os.path.join(os.path.dirname(__file__), 'roofline_plugin')
    os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\extras\CUPTI\lib64")
    os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\bin")
    os.add_dll_directory(dll_dir)

    dll_path = os.path.join(dll_dir, 'roofline_plugin.dll')
    print(f"Loading Plugin DLL from: {dll_path}")
    plugin = ctypes.CDLL(dll_path)

    # 2. Start True CUPTI Roofline Profiler & Enable Golden Zone Governor
    print("\n>>> STARTING TRUE CUPTI ROOFLINE PROFILER (AUTO-DISCOVERY) <<<")
    plugin.start_profiling_auto()
    time.sleep(1.0)

    print(">>> ENABLING DYNAMIC GOLDEN ZONE GOVERNOR (Memory: 945 MHz | Compute: 1950 MHz) <<<")
    plugin.enable_governor(945, 1950)
    time.sleep(1.0)

    # 3. Discover workloads
    compute_dir = os.path.join(os.path.dirname(__file__), 'workloads_KBENCH_EVAL', 'compute')
    memory_dir  = os.path.join(os.path.dirname(__file__), 'workloads_KBENCH_EVAL', 'memory')

    compute_files = glob.glob(os.path.join(compute_dir, "*.py"))
    memory_files  = glob.glob(os.path.join(memory_dir, "*.py"))

    print(f"Found {len(compute_files)} Compute workloads and {len(memory_files)} Memory workloads.")

    # 4. Alternate or randomly pick workloads for 6 rounds
    num_rounds = 20
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    for round_idx in range(1, num_rounds + 1):
        # Alternate between compute and memory, or pick randomly
        category = "compute" if round_idx % 2 == 1 else "memory"
        candidates = compute_files if category == "compute" else memory_files
        chosen_file = random.choice(candidates)
        filename = os.path.basename(chosen_file)

        print(f"\n================================================================================")
        print(f" ROUND {round_idx}/{num_rounds} | EXPECTED CATEGORY: [{category.upper()}-BOUND]")
        print(f" RUNNING WORKLOAD: {filename}")
        print(f"================================================================================")

        try:
            mod = load_workload_module(chosen_file)

            # Cap batch size to prevent OOM on 6GB VRAM
            if hasattr(mod, 'batch_size') and mod.batch_size > 512:
                mod.batch_size = 512
            if hasattr(mod, 'dim') and mod.dim > 131072:
                mod.dim = 131072

            init_inputs = mod.get_init_inputs() if hasattr(mod, 'get_init_inputs') else []
            model = mod.Model(*init_inputs).cuda()
            model.eval()

            inputs = [x.cuda() for x in mod.get_inputs()]

            # Run workload loop for 4.0 seconds
            t_start = time.time()
            iters = 0
            powers = []
            with torch.no_grad():
                while time.time() - t_start < 15.0:
                    model(*inputs)
                    torch.cuda.synchronize()
                    iters += 1
                    p = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                    powers.append(p)
                    time.sleep(0.01)

            avg_pwr = sum(powers) / len(powers) if powers else 0.0
            curr_clk = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"(Executed {iters} iterations | Clock: {curr_clk} MHz | Power: {avg_pwr:.1f} W)")

            # Clean up VRAM before next workload
            del model, inputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Error executing {filename}: {e}")

        time.sleep(0.5)

    print("\n================================================================================")
    print("Stopping Profiler & Releasing Clocks...")
    plugin.stop_profiling()
    time.sleep(1.0)
    final_clk = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
    print(f"Final Released Clock: {final_clk} MHz")
    print("Benchmark complete.")

if __name__ == '__main__':
    main()
