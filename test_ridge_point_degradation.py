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
    time.sleep(0.5)

def load_intensity_library():
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intensity_kernels.dll')
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"Missing compiled kernel library: {dll_path}")
    lib = ctypes.CDLL(dll_path)
    lib.launch_symmetric_memory_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.launch_ridge_point_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.launch_symmetric_compute_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return lib

def run_benchmark(lib, kernel_type, iterations):
    N = 32 * 1024 * 1024
    d_in = torch.randn(N, device='cuda', dtype=torch.float32)
    d_out = torch.empty_like(d_in)

    p_in = ctypes.c_void_p(d_in.data_ptr())
    p_out = ctypes.c_void_p(d_out.data_ptr())

    if kernel_type == 'memory':
        launcher = lib.launch_symmetric_memory_kernel
    elif kernel_type == 'ridge':
        launcher = lib.launch_ridge_point_kernel
    elif kernel_type == 'compute':
        launcher = lib.launch_symmetric_compute_kernel
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

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
    runtime = run_benchmark(lib, kernel_type, iterations=iterations)
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
    print("=" * 105)
    print(" EMPIRICAL ROOFLINE BENCHMARK: EXACT RIDGE POINT EVALUATION (I* = 56.79 FLOPs/Byte)")
    print(" Target GPU: NVIDIA GeForce RTX 4050 Laptop (AD107)")
    print(" Hardware Constants: P_peak = 10.91 TFLOP/s | B_peak = 192.02 GB/s | Ridge Point = 56.79 FLOPs/Byte")
    print("=" * 105)

    handle = get_nvml_handle()
    device_name = pynvml.nvmlDeviceGetName(handle)
    print(f"Active Device: {device_name}")

    lib = load_intensity_library()

    # Workloads to evaluate:
    # 1. Exactly at Ridge Point: I = 56.75 FLOPs/Byte (227 FMAs/float, Delta = -0.04 from I*)
    # 2. Equidistant Memory:     I = 26.75 FLOPs/Byte (107 FMAs/float, Delta = -30.04 from I*)
    # 3. Equidistant Compute:    I = 86.75 FLOPs/Byte (347 FMAs/float, Delta = +29.96 from I*)
    workloads = [
        ("Memory-Bound (Delta = -30.0)", 26.75, 'memory', 1000),
        ("EXACT Ridge Point (Delta = 0.0)", 56.75, 'ridge', 700),
        ("Compute-Bound (Delta = +30.0)", 86.75, 'compute', 500)
    ]

    caps_to_test = [
        (None, "Highest Cap (Unconstrained Boost)"),
        (945,  "Golden Zone Cap (945 MHz)"),
        (210,  "Lowest Hardware Cap (210 MHz)")
    ]

    all_results = []

    try:
        for w_name, intensity, k_type, iters in workloads:
            print("\n" + "#" * 105)
            print(f" WORKLOAD: {w_name.upper()} | I = {intensity:.2f} FLOPs/Byte | {iters} Iterations")
            print("#" * 105)
            for clock_val, label in caps_to_test:
                print(f"  >> Testing {label} [Target: {clock_val} MHz]...")
                res = benchmark_scenario(
                    handle, lib,
                    w_name, intensity, k_type,
                    clock_val, label,
                    iterations=iters
                )
                all_results.append(res)
                print(f"     Runtime : {res['runtime_s']:>7.3f} s")
                print(f"     Avg Clk : {res['avg_clock_mhz']:>7.1f} MHz")
                print(f"     Power   : {res['avg_power_w']:>7.1f} W")
                print(f"     Energy  : {res['energy_joules']:>7.1f} J")
                time.sleep(1.0)

    finally:
        print("\n>>> CLEANUP: RESETTING GPU LOCKED CLOCKS <<<")
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)

    # ---------------------------------------------------------------------
    # COMPARATIVE SUMMARY TABLE
    # ---------------------------------------------------------------------
    print("\n" + "=" * 135)
    print(f"{'Workload':<35} | {'Intensity':<12} | {'Clock Cap':<32} | {'Runtime':<9} | {'Avg Power':<9} | {'Energy':<9} | {'Perf Drop':<10} | {'Energy Saved':<12}")
    print("=" * 135)

    grouped = defaultdict(list)
    for r in all_results:
        grouped[r['workload']].append(r)

    for w_name, runs in grouped.items():
        baseline = runs[0]
        t_base = baseline['runtime_s']
        e_base = baseline['energy_joules']

        for r in runs:
            perf_drop = ((r['runtime_s'] - t_base) / t_base) * 100.0
            energy_saved = ((e_base - r['energy_joules']) / e_base) * 100.0

            perf_str = f"{perf_drop:>+7.2f} %" if r != baseline else "  BASELINE"
            energy_str = f"{energy_saved:>+7.2f} %" if r != baseline else "  BASELINE"

            print(f"{r['workload']:<35} | {r['intensity']:>5.2f} FLOP/B | {r['cap_label']:<32} | {r['runtime_s']:>7.3f}s | {r['avg_power_w']:>7.1f}W | {r['energy_joules']:>7.1f}J | {perf_str:<10} | {energy_str:<12}")
        print("-" * 135)

if __name__ == "__main__":
    main()
