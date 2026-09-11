import time
import statistics
import torch
import pynvml

def main():
    print("================================================================================")
    print(" BENCHMARK: GPU CLOCK-LOCKING OVERHEAD & SWITCHING LATENCY (DELTA_T)")
    print("================================================================================")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device_name = pynvml.nvmlDeviceGetName(handle)
    print(f"Device: {device_name}")

    target_clock_memory = 945 # Golden Zone clock for Memory-bound workloads

    # Warmup
    pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_clock_memory, target_clock_memory)
    pynvml.nvmlDeviceResetGpuLockedClocks(handle)

    # -------------------------------------------------------------------------
    # PART 1: Single Switch Latency Distribution (50 trials)
    # -------------------------------------------------------------------------
    print("\n[PART 1] Measuring Single Switch Latency across 50 trials...")
    lock_times_ms = []
    reset_times_ms = []

    for _ in range(50):
        # Time Lock call
        t0 = time.perf_counter()
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_clock_memory, target_clock_memory)
        t1 = time.perf_counter()
        lock_times_ms.append((t1 - t0) * 1000.0)

        # Brief rest
        time.sleep(0.01)

        # Time Reset call
        t2 = time.perf_counter()
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        t3 = time.perf_counter()
        reset_times_ms.append((t3 - t2) * 1000.0)

        time.sleep(0.01)

    print("\n--- Single Switch Latency Results (ms) ---")
    print(f"Lock Call (-> 945 MHz) : Mean: {statistics.mean(lock_times_ms):.3f} ms | "
          f"Median: {statistics.median(lock_times_ms):.3f} ms | "
          f"Min: {min(lock_times_ms):.3f} ms | Max: {max(lock_times_ms):.3f} ms | "
          f"StdDev: {statistics.stdev(lock_times_ms):.3f} ms")

    print(f"Reset Call (-> Default): Mean: {statistics.mean(reset_times_ms):.3f} ms | "
          f"Median: {statistics.median(reset_times_ms):.3f} ms | "
          f"Min: {min(reset_times_ms):.3f} ms | Max: {max(reset_times_ms):.3f} ms | "
          f"StdDev: {statistics.stdev(reset_times_ms):.3f} ms")

    avg_single_switch_ms = (statistics.mean(lock_times_ms) + statistics.mean(reset_times_ms)) / 2.0
    print(f"\nAverage Single Transition Overhead (delta_t_switch): {avg_single_switch_ms:.3f} ms")

    # -------------------------------------------------------------------------
    # PART 2: Multi-Switch Linearity Scaling Test (1x, 2x, 4x, 5x, 10x)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("[PART 2] Testing Multi-Switch Scaling Linearity (1x, 2x, 4x, 5x, 10x)...")
    print("================================================================================")

    switch_counts = [1, 2, 4, 5, 10]
    multi_results = {}

    for count in switch_counts:
        trials = []
        for _ in range(10): # 10 trials per count
            t0 = time.perf_counter()
            for i in range(count):
                if i % 2 == 0:
                    pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_clock_memory, target_clock_memory)
                else:
                    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            t1 = time.perf_counter()
            # Clean up state
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            trials.append((t1 - t0) * 1000.0)

        mean_total_ms = statistics.mean(trials)
        mean_per_switch_ms = mean_total_ms / count
        multi_results[count] = (mean_total_ms, mean_per_switch_ms)

        print(f"  {count:2d} Switches: Total Time = {mean_total_ms:7.3f} ms | Per-Switch = {mean_per_switch_ms:6.3f} ms")

    # Linearity check: compare ratio to 1 switch
    baseline_1 = multi_results[1][0]
    print("\n--- Linearity Analysis ---")
    for count in switch_counts[1:]:
        actual = multi_results[count][0]
        expected = baseline_1 * count
        ratio = actual / expected
        print(f"  {count}x Switches: Actual = {actual:6.2f} ms | Expected (Linear) = {expected:6.2f} ms | Linearity Ratio = {ratio:.2f}x")

    # -------------------------------------------------------------------------
    # PART 3: In-Flight GPU Execution Stall Test (Impact on PyTorch kernels)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("[PART 3] Testing Impact of Switching Clock Caps While GPU is Running PyTorch")
    print("================================================================================")

    # Initialize PyTorch CUDA
    a = torch.randn(2048, 2048, device='cuda')
    b = torch.randn(2048, 2048, device='cuda')
    torch.cuda.synchronize()

    # Measure kernel iteration time WITHOUT switching (Baseline)
    baseline_kernel_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline_kernel_times.append((t1 - t0) * 1000.0)

    # Measure kernel iteration time WITH a clock switch occurring during execution
    switching_kernel_times = []
    for i in range(50):
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        if i % 2 == 0:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_clock_memory, target_clock_memory)
        else:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        switching_kernel_times.append((t1 - t0) * 1000.0)

    # Reset
    pynvml.nvmlDeviceResetGpuLockedClocks(handle)

    mean_baseline = statistics.mean(baseline_kernel_times)
    mean_switching = statistics.mean(switching_kernel_times)

    print(f"Baseline Kernel Iteration Time (No Switching): {mean_baseline:.3f} ms")
    print(f"Kernel Iteration Time WITH Clock Switching   : {mean_switching:.3f} ms")
    print(f"Kernel Execution Overhead from Switch        : {mean_switching - mean_baseline:.3f} ms")

    print("\n================================================================================")
    print(" BENCHMARK COMPLETE.")
    print("================================================================================")

if __name__ == '__main__':
    main()
