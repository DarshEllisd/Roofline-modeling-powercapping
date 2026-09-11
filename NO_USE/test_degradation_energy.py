import os
import sys
import time
import threading
from collections import defaultdict
import torch
import pynvml

def get_nvml_handle():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)

class PrecisionPowerMonitor:
    """
    Samples NVML power and graphics clock continuously at ~10ms intervals.
    Calculates numerical energy integral: E = sum(P_i * dt_i).
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
    """Sets GPU locked graphics clocks, or resets to default boost if None."""
    if clock_mhz is None:
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    else:
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, clock_mhz, clock_mhz)
    time.sleep(0.5)  # Allow voltage/clock regulator to settle

def run_memory_workload(iterations=2000):
    """
    Workload 1: Memory-Bound (AI << Ridge Point).
    Tensor: 32M FP32 elements = 128 MB.
    Operation: In-place Tanh with pre-allocated output buffer.
    Memory traffic: 128 MB Read + 128 MB Write = 256 MB per iteration.
    Arithmetic Intensity: ~0.125 FLOPs/Byte (<< 56.79 Ridge Point).
    """
    x = torch.randn(32 * 1024 * 1024, device='cuda', dtype=torch.float32)
    out = torch.empty_like(x)

    # Warmup
    for _ in range(50):
        torch.tanh(x, out=out)
    torch.cuda.synchronize()

    # Benchmark loop (no allocations in loop, saturating memory bus)
    t0 = time.perf_counter()
    for _ in range(iterations):
        torch.tanh(x, out=out)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del x, out
    torch.cuda.empty_cache()
    return elapsed

def run_compute_workload(iterations=120):
    """
    Workload 2: Compute-Bound (AI >> Ridge Point).
    Dense FP32 GEMM: 4096 x 4096.
    FLOPs: 2 * 4096^3 = 137.44 GFLOPs per matmul.
    Arithmetic Intensity: ~680 FLOPs/Byte (>> 56.79 Ridge Point).
    """
    a = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    b = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    c = torch.empty(4096, 4096, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(5):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    # Benchmark loop
    t0 = time.perf_counter()
    for _ in range(iterations):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del a, b, c
    torch.cuda.empty_cache()
    return elapsed

def benchmark_scenario(handle, workload_name, workload_fn, clock_cap, cap_label, iterations):
    set_gpu_clock_cap(handle, clock_cap)
    monitor = PrecisionPowerMonitor(handle, poll_interval_s=0.01)

    monitor.start()
    runtime = workload_fn(iterations)
    avg_power, avg_clock, integrated_energy = monitor.stop()

    # If trapezoidal integral is available use it, otherwise avg_power * runtime
    energy_joules = integrated_energy if integrated_energy > 0 else (avg_power * runtime)

    return {
        'workload': workload_name,
        'cap_label': cap_label,
        'clock_target': clock_cap if clock_cap is not None else 'Unconstrained (Boost)',
        'avg_clock_mhz': avg_clock,
        'runtime_s': runtime,
        'avg_power_w': avg_power,
        'energy_joules': energy_joules
    }

def main():
    print("=" * 85)
    print(" EMPIRICAL ROOFLINE DEGRADATION & ENERGY BENCHMARK")
    print(" Hardware: NVIDIA GeForce RTX 4050 Laptop (AD107)")
    print(" Hardware Ridge Point I* = 56.79 FLOPs/Byte")
    print("=" * 85)

    handle = get_nvml_handle()
    device_name = pynvml.nvmlDeviceGetName(handle)
    print(f"Active Device: {device_name}")

    # Testing regimes:
    # 1. Highest Cap: Unconstrained dynamic boost (No cap, ~2415+ MHz)
    # 2. Golden Zone Cap: 945 MHz (Discovered minimum efficient operating point)
    # 3. Lowest Hardware Cap: 210 MHz (Absolute minimum hardware clock supported by AD107)
    caps_to_test = [
        (None, "Highest Cap (Unconstrained Boost)"),
        (945,  "Golden Zone Cap (945 MHz)"),
        (210,  "Lowest Hardware Cap (210 MHz)")
    ]

    all_results = []

    try:
        # =====================================================================
        # WORKLOAD 1: MEMORY-BOUND (AI = 0.125 FLOPs/Byte << Ridge Point 56.79)
        # =====================================================================
        print("\n" + "#" * 85)
        print(" WORKLOAD 1: MEMORY-BOUND (AI ~ 0.125 FLOPs/Byte << Ridge Point 56.79)")
        print(" Kernel: In-place Tanh on 32M Floats (128 MB VRAM, 256 MB traffic/iter)")
        print("#" * 85)
        for clock_val, label in caps_to_test:
            print(f"\n>> Testing {label} [Target: {clock_val} MHz]...")
            res = benchmark_scenario(
                handle,
                "Memory-Bound (Tanh 128MB)",
                run_memory_workload,
                clock_val,
                label,
                iterations=2000
            )
            all_results.append(res)
            print(f"   Runtime : {res['runtime_s']:>7.3f} s")
            print(f"   Avg Clk : {res['avg_clock_mhz']:>7.1f} MHz")
            print(f"   Power   : {res['avg_power_w']:>7.1f} W")
            print(f"   Energy  : {res['energy_joules']:>7.1f} J")
            time.sleep(1.0)

        # =====================================================================
        # WORKLOAD 2: COMPUTE-BOUND (AI = ~680 FLOPs/Byte >> Ridge Point 56.79)
        # =====================================================================
        print("\n" + "#" * 85)
        print(" WORKLOAD 2: COMPUTE-BOUND (AI ~ 680 FLOPs/Byte >> Ridge Point 56.79)")
        print(" Kernel: Dense FP32 GEMM 4096 x 4096 (137.4 GFLOPs/iter)")
        print("#" * 85)
        for clock_val, label in caps_to_test:
            print(f"\n>> Testing {label} [Target: {clock_val} MHz]...")
            # For 210 MHz, 120 iterations might take ~20s; let's use 80 iterations
            res = benchmark_scenario(
                handle,
                "Compute-Bound (Dense GEMM 4096)",
                run_compute_workload,
                clock_val,
                label,
                iterations=80
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

    # =====================================================================
    # COMPARATIVE SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 115)
    print(f"{'Workload':<30} | {'Clock Cap':<32} | {'Runtime':<10} | {'Avg Power':<10} | {'Energy':<10} | {'Perf Drop':<10} | {'Energy Saved':<12}")
    print("=" * 115)

    grouped = defaultdict(list)
    for r in all_results:
        grouped[r['workload']].append(r)

    for w_name, runs in grouped.items():
        baseline = runs[0]  # Highest Cap is the reference baseline
        t_base = baseline['runtime_s']
        e_base = baseline['energy_joules']

        for r in runs:
            perf_drop = ((r['runtime_s'] - t_base) / t_base) * 100.0
            energy_saved = ((e_base - r['energy_joules']) / e_base) * 100.0

            perf_str = f"{perf_drop:>+7.2f} %" if r != baseline else "  BASELINE"
            energy_str = f"{energy_saved:>+7.2f} %" if r != baseline else "  BASELINE"

            print(f"{r['workload']:<30} | {r['cap_label']:<32} | {r['runtime_s']:>8.3f}s | {r['avg_power_w']:>8.1f}W | {r['energy_joules']:>8.1f}J | {perf_str:<10} | {energy_str:<12}")
        print("-" * 115)

if __name__ == "__main__":
    main()
