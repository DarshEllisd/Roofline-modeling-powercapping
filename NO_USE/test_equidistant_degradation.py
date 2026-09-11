import os
import sys
import time
import ctypes
import threading
from collections import defaultdict
import torch
import pynvml

def get_nvml_handle():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)

class PrecisionPowerMonitor:
    """
    Continuously samples NVML power and graphics clock at ~10ms intervals.
    Calculates exact numerical energy integral via trapezoidal Riemann sum:
    E = sum_i 0.5 * (P_i + P_{i-1}) * (t_i - t_{i-1})
    """
    def __init__(self, handle, poll_interval_s=0.01):
        self.handle = handle
        self.poll_interval = poll_interval_s
        self.samples = []  # list of (timestamp, power_w, clock_mhz)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            t = time.perf_counter()
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                p_w = p_mw / 1000.0
                clk = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                self.samples.append((t, p_w, clk))
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        
        if len(self.samples) < 2:
            avg_pwr = self.samples[0][1] if self.samples else 0.0
            avg_clk = self.samples[0][2] if self.samples else 0.0
            return avg_pwr, avg_clk, 0.0

        # Numerical integration: trapezoidal rule
        total_energy_j = 0.0
        for i in range(1, len(self.samples)):
            t_prev, p_prev, _ = self.samples[i - 1]
            t_curr, p_curr, _ = self.samples[i]
            dt = t_curr - t_prev
            total_energy_j += 0.5 * (p_prev + p_curr) * dt

        avg_pwr = sum(s[1] for s in self.samples) / len(self.samples)
        avg_clk = sum(s[2] for s in self.samples) / len(self.samples)
        return avg_pwr, avg_clk, total_energy_j

def set_gpu_clock_cap(handle, clock_mhz):
    """Sets GPU locked graphics clocks, or resets to default dynamic boost if None."""
    if clock_mhz is None:
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    else:
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, clock_mhz, clock_mhz)
    time.sleep(0.5)  # Allow voltage/clock regulator to settle

def load_intensity_library():
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intensity_kernels.dll')
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"Missing compiled kernel library: {dll_path}")
    lib = ctypes.CDLL(dll_path)
    lib.launch_symmetric_memory_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.launch_symmetric_compute_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return lib

def run_equidistant_benchmark(lib, kernel_type, iterations=1000):
    """
    Executes either the symmetric memory kernel or symmetric compute kernel.
    Buffer size: 32M FP32 elements = 128 MB input, 128 MB output.
    Total DRAM traffic per pass = 256 MB (vastly exceeds 32 MB L2 cache).
    """
    N = 32 * 1024 * 1024
    d_in = torch.randn(N, device='cuda', dtype=torch.float32)
    d_out = torch.empty_like(d_in)

    p_in = ctypes.c_void_p(d_in.data_ptr())
    p_out = ctypes.c_void_p(d_out.data_ptr())

    launcher = lib.launch_symmetric_memory_kernel if kernel_type == 'memory' else lib.launch_symmetric_compute_kernel

    # Warmup
    for _ in range(20):
        launcher(p_out, p_in, N, None)
    torch.cuda.synchronize()

    # Benchmark loop
    t0 = time.perf_counter()
    for _ in range(iterations):
        launcher(p_out, p_in, N, None)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del d_in, d_out
    torch.cuda.empty_cache()
    return elapsed

def benchmark_scenario(handle, lib, workload_name, intensity_val, kernel_type, clock_cap, cap_label, iterations):
    set_gpu_clock_cap(handle, clock_cap)
    monitor = PrecisionPowerMonitor(handle, poll_interval_s=0.01)

    monitor.start()
    runtime = run_equidistant_benchmark(lib, kernel_type, iterations=iterations)
    avg_power, avg_clock, integrated_energy = monitor.stop()

    energy_joules = integrated_energy if integrated_energy > 0 else (avg_power * runtime)

    return {
        'workload': workload_name,
        'intensity': intensity_val,
        'cap_label': cap_label,
        'clock_target': clock_cap if clock_cap is not None else 'Unconstrained (Boost)',
        'avg_clock_mhz': avg_clock,
        'runtime_s': runtime,
        'avg_power_w': avg_power,
        'energy_joules': energy_joules
    }

def main():
    print("=" * 95)
    print(" EMPIRICAL EVALUATION: EQUIDISTANT WORKLOADS FROM HARDWARE RIDGE POINT")
    print(" Target GPU: NVIDIA GeForce RTX 4050 Laptop (AD107)")
    print(" Hardware Ridge Point I* = 56.79 FLOPs/Byte (P_peak: 10.91 TF, B_peak: 192.02 GB/s)")
    print(" Distance Delta |I - I*| = 30.0 FLOPs/Byte")
    print("=" * 95)

    handle = get_nvml_handle()
    device_name = pynvml.nvmlDeviceGetName(handle)
    print(f"Active Device: {device_name}")

    lib = load_intensity_library()

    # Define the Symmetric Pair equidistant from I* = 56.79:
    # 1. Memory-Bound Workload: I = 26.75 FLOPs/Byte  (Delta = -30.04 FLOPs/Byte)
    #    107 FMAs / float = 214 FLOPs / 8 Bytes = 26.75 FLOPs/Byte
    # 2. Compute-Bound Workload: I = 86.75 FLOPs/Byte (Delta = +29.96 FLOPs/Byte)
    #    347 FMAs / float = 694 FLOPs / 8 Bytes = 86.75 FLOPs/Byte
    
    caps_to_test = [
        (None, "Highest Cap (Unconstrained Boost)"),
        (945,  "Golden Zone Cap (945 MHz)"),
        (210,  "Lowest Hardware Cap (210 MHz)")
    ]

    all_results = []

    try:
        # ---------------------------------------------------------------------
        # WORKLOAD 1: EQUIDISTANT MEMORY-BOUND (I = 26.75, Delta = -30.04)
        # ---------------------------------------------------------------------
        print("\n" + "#" * 95)
        print(" WORKLOAD 1: EQUIDISTANT MEMORY-BOUND (I = 26.75 FLOPs/Byte | Delta = -30.04 below I*)")
        print(" Kernel: 107 FMAs per float on 32M Floats (128 MB read + 128 MB write = 256 MB traffic)")
        print("#" * 95)
        for clock_val, label in caps_to_test:
            print(f"\n>> Testing {label} [Target: {clock_val} MHz]...")
            res = benchmark_scenario(
                handle, lib,
                "Memory-Bound (I = 26.75, -30.0 from I*)",
                26.75, 'memory',
                clock_val, label,
                iterations=1000
            )
            all_results.append(res)
            print(f"   Runtime : {res['runtime_s']:>7.3f} s")
            print(f"   Avg Clk : {res['avg_clock_mhz']:>7.1f} MHz")
            print(f"   Power   : {res['avg_power_w']:>7.1f} W")
            print(f"   Energy  : {res['energy_joules']:>7.1f} J")
            time.sleep(1.0)

        # ---------------------------------------------------------------------
        # WORKLOAD 2: EQUIDISTANT COMPUTE-BOUND (I = 86.75, Delta = +29.96)
        # ---------------------------------------------------------------------
        print("\n" + "#" * 95)
        print(" WORKLOAD 2: EQUIDISTANT COMPUTE-BOUND (I = 86.75 FLOPs/Byte | Delta = +29.96 above I*)")
        print(" Kernel: 347 FMAs per float on 32M Floats (128 MB read + 128 MB write = 256 MB traffic)")
        print("#" * 95)
        for clock_val, label in caps_to_test:
            print(f"\n>> Testing {label} [Target: {clock_val} MHz]...")
            # For 210 MHz, 500 iterations is ~11s
            res = benchmark_scenario(
                handle, lib,
                "Compute-Bound (I = 86.75, +30.0 from I*)",
                86.75, 'compute',
                clock_val, label,
                iterations=500
            )
            all_results.append(res)
            print(f"   Runtime : {res['runtime_s']:>7.3f} s")
            print(f"   Avg Clk : {res['avg_clock_mhz']:>7.1f} MHz")
            print(f"   Power   : {res['avg_power_w']:>7.1f} W")
            print(f"   Energy  : {res['energy_joules']:>7.1f} J")
            time.sleep(1.0)

    finally:
        print("\n>>> CLEANUP: RESETTING GPU LOCKED CLOCKS <<<")
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)

    # ---------------------------------------------------------------------
    # COMPARATIVE SUMMARY TABLE
    # ---------------------------------------------------------------------
    print("\n" + "=" * 125)
    print(f"{'Workload':<42} | {'Clock Cap':<32} | {'Runtime':<10} | {'Avg Power':<10} | {'Energy':<10} | {'Perf Drop':<10} | {'Energy Saved':<12}")
    print("=" * 125)

    grouped = defaultdict(list)
    for r in all_results:
        grouped[r['workload']].append(r)

    for w_name, runs in grouped.items():
        baseline = runs[0]  # Highest Cap is baseline
        t_base = baseline['runtime_s']
        e_base = baseline['energy_joules']

        for r in runs:
            perf_drop = ((r['runtime_s'] - t_base) / t_base) * 100.0
            energy_saved = ((e_base - r['energy_joules']) / e_base) * 100.0

            perf_str = f"{perf_drop:>+7.2f} %" if r != baseline else "  BASELINE"
            energy_str = f"{energy_saved:>+7.2f} %" if r != baseline else "  BASELINE"

            print(f"{r['workload']:<42} | {r['cap_label']:<32} | {r['runtime_s']:>8.3f}s | {r['avg_power_w']:>8.1f}W | {r['energy_joules']:>8.1f}J | {perf_str:<10} | {energy_str:<12}")
        print("-" * 125)

if __name__ == "__main__":
    main()
