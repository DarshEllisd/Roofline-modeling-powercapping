#!/usr/bin/env python3
"""
Hierarchical Real-Time Roofline Golden Zone Explorer (DRAM vs. L2 Cache)
========================================================================
Sweeps Arithmetic Intensity (AI) from ~0 upwards in intervals of 10 FLOPs/Byte
for both:
  1. GDDR DRAM (External Memory Bus, Working Set = 256 MB >> 64 MB L2 Cache)
     Theoretical Ridge Point: I*_DRAM = 65,280 GFLOP/s / 576 GB/s ~ 113.3 FLOPs/Byte
  2. L2 Cache (On-Chip Crossbar, Working Set = 16 MB << 64 MB L2 Cache)
     Theoretical Ridge Point: I*_L2 = 65,280 GFLOP/s / ~2,600 GB/s ~ 25.1 FLOPs/Byte

For each memory regime and each AI, sweeps GPU Power Caps [250W down to 100W] on RTX 5000 Ada.
Measures:
  - Kernel execution runtime (seconds via CUDA Events)
  - Power draw (Watts sampled at 50Hz via NVML)
  - Total Energy consumed (Joules = Power * Time)
  - Performance degradation vs unconstrained baseline (%)
  - Net energy savings (%)
  - Achieved memory bandwidth (GB/s) & achieved compute (TFLOP/s)
  - Discovers the Pareto-optimal Golden Zone Power Cap for each AI
  - Evaluates Golden Zone saturation across both memory hierarchies
"""

import os
import sys
import time
import ctypes
import argparse
import threading
import subprocess
import numpy as np
import pandas as pd
import torch
import pynvml

class PowerTelemetry(threading.Thread):
    """High-frequency background thread sampling NVML power and clock rates."""
    def __init__(self, handle, sample_interval_s=0.015):
        super().__init__()
        self.handle = handle
        self.sample_interval_s = sample_interval_s
        self.stop_event = threading.Event()
        self.power_samples = []
        self.clock_samples = []
        self.temp_samples = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_samples.append(p_mw / 1000.0)
                clk = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                self.clock_samples.append(clk)
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                self.temp_samples.append(temp)
            except Exception:
                pass
            time.sleep(self.sample_interval_s)

    def stop(self):
        self.stop_event.set()
        self.join()
        avg_power = float(np.mean(self.power_samples)) if self.power_samples else 0.0
        peak_power = float(np.max(self.power_samples)) if self.power_samples else 0.0
        min_power = float(np.min(self.power_samples)) if self.power_samples else 0.0
        avg_clock = float(np.mean(self.clock_samples)) if self.clock_samples else 0.0
        avg_temp = float(np.mean(self.temp_samples)) if self.temp_samples else 0.0
        return {
            'avg_power_w': avg_power,
            'peak_power_w': peak_power,
            'min_power_w': min_power,
            'avg_clock_mhz': avg_clock,
            'avg_temp_c': avg_temp,
            'sample_count': len(self.power_samples)
        }

def set_gpu_power_cap(handle, cap_watts):
    """Applies power cap via NVML or fallback to sudo nvidia-smi."""
    cap_mw = int(cap_watts * 1000)
    try:
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, cap_mw)
        return True, "NVML direct"
    except pynvml.NVMLError_NoPermission:
        try:
            cmd = ["sudo", "-n", "nvidia-smi", "-pl", str(int(cap_watts))]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                return True, "sudo nvidia-smi"
        except Exception:
            pass
        return False, "Permission Denied: Run with sudo or setcap to alter power caps"
    except Exception as e:
        return False, str(e)

def reset_gpu_power_cap(handle, default_watts):
    set_gpu_power_cap(handle, default_watts)

def load_kernel_library(lib_path):
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Kernel library not found: {lib_path}. Run compile_kernels.sh first!")
    lib = ctypes.CDLL(lib_path)
    lib.launch_intensity_benchmark.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
    ]
    lib.launch_intensity_benchmark.restype = None

    lib.get_suggested_fma_count.argtypes = [ctypes.c_double]
    lib.get_suggested_fma_count.restype = ctypes.c_int
    return lib

def run_single_regime(regime_name, buffer_mb, iterations, warmup_iters,
                       args, nvml_handle, kernel_lib, stream, device):
    """Executes the AI & Power Cap sweep for a specific memory hierarchy (DRAM or L2)."""
    print(f"\n{'='*80}")
    print(f"STARTING REGIME: {regime_name.upper()}")
    print(f"Working Set Buffer: {buffer_mb} MB")
    print(f"Iterations per Run: {iterations} (Warmup: {warmup_iters})")
    print(f"{'='*80}")

    num_elements = (buffer_mb * 1024 * 1024) // 4
    d_in = torch.randn(num_elements, dtype=torch.float32, device=device)
    d_out = torch.zeros(num_elements, dtype=torch.float32, device=device)
    bytes_per_iteration = num_elements * 8

    # Generate AI range
    ai_values = []
    curr_ai = args.ai_min
    while curr_ai <= args.ai_max + 1e-5:
        ai_values.append(round(curr_ai, 2))
        curr_ai += args.ai_step

    regime_dir = os.path.join(args.output_dir, regime_name.lower())
    os.makedirs(regime_dir, exist_ok=True)

    raw_results = []
    golden_zone_summary = []
    consecutive_saturated = 0

    for ai in ai_values:
        n_fma = kernel_lib.get_suggested_fma_count(ctypes.c_double(ai))
        achieved_ai = (2.0 * n_fma) / 8.0 if n_fma > 0 else 0.0
        flops_per_iteration = num_elements * (2 * n_fma)

        print(f"\n[{regime_name.upper()}] AI Target: {ai:.1f} FLOP/B | Kernel FMA: {n_fma} | True AI: {achieved_ai:.2f} FLOP/B")
        print(f"{'-'*75}")

        baseline_runtime = None
        baseline_energy = None
        baseline_power = None
        ai_run_data = []

        # Calibrate iteration count to sustain ~0.8s per pass for reliable NVML sampling
        kernel_lib.launch_intensity_benchmark(
            ctypes.c_void_p(d_out.data_ptr()),
            ctypes.c_void_p(d_in.data_ptr()),
            ctypes.c_int(num_elements),
            ctypes.c_int(n_fma),
            ctypes.c_int(5),
            ctypes.c_void_p(stream)
        )
        torch.cuda.synchronize()
        t_cal0 = time.perf_counter()
        kernel_lib.launch_intensity_benchmark(
            ctypes.c_void_p(d_out.data_ptr()),
            ctypes.c_void_p(d_in.data_ptr()),
            ctypes.c_int(num_elements),
            ctypes.c_int(n_fma),
            ctypes.c_int(10),
            ctypes.c_void_p(stream)
        )
        torch.cuda.synchronize()
        t_10 = max(time.perf_counter() - t_cal0, 1e-5)
        active_iters = max(10, int(10.0 * (args.target_duration_s / t_10)))

        for cap in args.power_caps:
            if not args.dry_run:
                set_gpu_power_cap(nvml_handle, cap)
            time.sleep(0.12)

            # Warmup pass (critical for L2 cache residency to populate lines)
            kernel_lib.launch_intensity_benchmark(
                ctypes.c_void_p(d_out.data_ptr()),
                ctypes.c_void_p(d_in.data_ptr()),
                ctypes.c_int(num_elements),
                ctypes.c_int(n_fma),
                ctypes.c_int(warmup_iters),
                ctypes.c_void_p(stream)
            )
            torch.cuda.synchronize()

            # Timed Execution with Power Telemetry
            telemetry = PowerTelemetry(nvml_handle, sample_interval_s=0.01)
            telemetry.start()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            kernel_lib.launch_intensity_benchmark(
                ctypes.c_void_p(d_out.data_ptr()),
                ctypes.c_void_p(d_in.data_ptr()),
                ctypes.c_int(num_elements),
                ctypes.c_int(n_fma),
                ctypes.c_int(active_iters),
                ctypes.c_void_p(stream)
            )
            end_event.record()
            torch.cuda.synchronize()

            metrics = telemetry.stop()
            runtime_s = start_event.elapsed_time(end_event) / 1000.0

            avg_power_w = metrics['avg_power_w']
            total_energy_j = avg_power_w * runtime_s

            total_bytes = bytes_per_iteration * active_iters
            total_flops = flops_per_iteration * active_iters
            achieved_bw_gbs = (total_bytes / runtime_s) / 1e9
            achieved_tflops = (total_flops / runtime_s) / 1e12

            if cap == args.power_caps[0]:
                baseline_runtime = runtime_s
                baseline_energy = total_energy_j
                baseline_power = avg_power_w
                deg_pct = 0.0
                energy_saved_pct = 0.0
                energy_saved_j = 0.0
            else:
                deg_pct = ((runtime_s / baseline_runtime) - 1.0) * 100.0
                energy_saved_j = baseline_energy - total_energy_j
                energy_saved_pct = (energy_saved_j / baseline_energy) * 100.0 if baseline_energy > 0 else 0.0

            trial_record = {
                'regime': regime_name,
                'target_ai': ai,
                'kernel_fma': n_fma,
                'true_ai': achieved_ai,
                'power_cap_w': cap,
                'runtime_s': runtime_s,
                'avg_power_w': avg_power_w,
                'peak_power_w': metrics['peak_power_w'],
                'avg_clock_mhz': metrics['avg_clock_mhz'],
                'avg_temp_c': metrics['avg_temp_c'],
                'total_energy_j': total_energy_j,
                'perf_deg_pct': deg_pct,
                'energy_saved_j': energy_saved_j,
                'energy_saved_pct': energy_saved_pct,
                'achieved_bw_gbs': achieved_bw_gbs,
                'achieved_tflops': achieved_tflops
            }
            raw_results.append(trial_record)
            ai_run_data.append(trial_record)

            print(f"  [{regime_name}] Cap {cap:>3}W: Time={runtime_s:.4f}s | Power={avg_power_w:>5.1f}W | Energy={total_energy_j:>6.1f}J | Deg={deg_pct:>+6.2f}% | Saved={energy_saved_pct:>+6.2f}% | BW={achieved_bw_gbs:>6.1f}GB/s")

        # Determine Golden Zone Cap:
        # Must satisfy performance degradation <= deg_threshold AND have positive energy savings (> 0.0%)
        eligible_caps = [r for r in ai_run_data if r['perf_deg_pct'] <= args.deg_threshold and r['energy_saved_pct'] > 0.0]
        if eligible_caps:
            # Select the cap maximizing net energy saved
            eligible_caps.sort(key=lambda x: x['energy_saved_pct'], reverse=True)
            best_gz = eligible_caps[0]
        else:
            # If no reduced cap saves energy within degradation budget, Golden Zone is baseline unconstrained cap
            best_gz = ai_run_data[0]
        is_saturated = (best_gz['power_cap_w'] == args.power_caps[0])

        gz_record = {
            'regime': regime_name,
            'target_ai': ai,
            'true_ai': achieved_ai,
            'golden_cap_w': best_gz['power_cap_w'],
            'golden_runtime_s': best_gz['runtime_s'],
            'golden_power_w': best_gz['avg_power_w'],
            'golden_energy_j': best_gz['total_energy_j'],
            'golden_deg_pct': best_gz['perf_deg_pct'],
            'golden_energy_saved_pct': best_gz['energy_saved_pct'],
            'achieved_bw_gbs': best_gz['achieved_bw_gbs'],
            'baseline_cap_w': args.power_caps[0],
            'baseline_runtime_s': baseline_runtime,
            'baseline_power_w': baseline_power,
            'baseline_energy_j': baseline_energy,
            'is_saturated': is_saturated
        }
        golden_zone_summary.append(gz_record)

        print(f"  >> [{regime_name.upper()} RESULT for AI={ai:.1f}]: Golden Cap = {best_gz['power_cap_w']} W (Perf Drop: {best_gz['perf_deg_pct']:.2f}%, Energy Saved: {best_gz['energy_saved_pct']:.2f}%)")

        if is_saturated:
            consecutive_saturated += 1
            print(f"  >> [{regime_name.upper()} SATURATION] ({consecutive_saturated}/{args.saturation_count}) Golden Zone saturated at baseline {args.power_caps[0]}W.")
            if consecutive_saturated >= args.saturation_count:
                print(f"\n[EARLY STOPPING {regime_name.upper()}] Saturated for {args.saturation_count} consecutive AI steps. Ending regime.")
                break
        else:
            consecutive_saturated = 0

    # Save Regime CSVs
    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(regime_dir, "raw_sweep_data.csv"), index=False)

    gz_df = pd.DataFrame(golden_zone_summary)
    gz_df.to_csv(os.path.join(regime_dir, "golden_zone_by_ai.csv"), index=False)
    print(f"\nSaved {regime_name} datasets to: {regime_dir}")
    return raw_df, gz_df

def generate_comparative_report(dram_gz, l2_gz, output_dir, threshold):
    """Produces the comprehensive comparative report comparing GDDR DRAM vs L2 Cache Rooflines."""
    report_path = os.path.join(output_dir, "hierarchical_roofline_summary.md")
    with open(report_path, "w") as f:
        f.write("# Hierarchical Real-Time Roofline: GDDR DRAM vs. L2 Cache Golden Zones\n\n")
        f.write("**Target Hardware:** NVIDIA RTX 5000 Ada Generation (AD102, CC 8.9)\n")
        f.write(f"**Tolerance Threshold:** $\\le {threshold:.1f}\\%$ Performance Degradation\n\n")

        f.write("## 1. Architectural Constants & Ridge Point Divergence\n\n")
        f.write("| Parameter | GDDR6 DRAM | L2 Cache (On-Chip Crossbar) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| **Theoretical Peak Bandwidth ($B_{\\text{peak}}$)** | **$576.0\\text{ GB/s}$** | **$\\sim 2,600.0\\text{ GB/s}$** ($\\approx 4.5\\times$ higher) |\n")
        f.write("| **Compute Peak ($P_{\\text{peak}}$)** | $65.28\\text{ TFLOP/s}$ | $65.28\\text{ TFLOP/s}$ |\n")
        f.write("| **Hardware Ridge Point ($I^*$)** | **$113.33\\text{ FLOPs/Byte}$** | **$\\approx 25.10\\text{ FLOPs/Byte}$** |\n")
        f.write("| **Working Set Tested** | $256\\text{ MB}$ (Flushes $64\\text{ MB}$ L2 cache) | $16\\text{ MB}$ (Resides $100\\%$ inside L2 cache) |\n\n")

        f.write("## 2. Side-by-Side Golden Zone Comparison\n\n")
        f.write("| Arithmetic Intensity | GDDR DRAM Golden Cap | DRAM Energy Saved | L2 Cache Golden Cap | L2 Energy Saved | Architectural Divergence |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

        # Outer join or unified AI index
        all_ais = sorted(list(set(
            (dram_gz['target_ai'].tolist() if dram_gz is not None else []) +
            (l2_gz['target_ai'].tolist() if l2_gz is not None else [])
        )))

        dram_map = {r['target_ai']: r for _, r in dram_gz.iterrows()} if dram_gz is not None else {}
        l2_map = {r['target_ai']: r for _, r in l2_gz.iterrows()} if l2_gz is not None else {}

        for ai in all_ais:
            d_row = dram_map.get(ai, None)
            l_row = l2_map.get(ai, None)

            d_cap_str = f"**{d_row['golden_cap_w']} W**" if d_row is not None else "N/A"
            d_sav_str = f"{d_row['golden_energy_saved_pct']:>+5.2f} %" if d_row is not None else "N/A"
            
            l_cap_str = f"**{l_row['golden_cap_w']} W**" if l_row is not None else "N/A"
            l_sav_str = f"{l_row['golden_energy_saved_pct']:>+5.2f} %" if l_row is not None else "N/A"

            if d_row is not None and l_row is not None:
                if d_row['golden_cap_w'] == l_row['golden_cap_w']:
                    div_str = "Identical Cap"
                elif d_row['golden_cap_w'] < l_row['golden_cap_w']:
                    div_str = f"L2 Saturates Earlier (+{l_row['golden_cap_w'] - d_row['golden_cap_w']}W)"
                else:
                    div_str = "DRAM Higher"
            else:
                div_str = "Saturated / Ended"

            f.write(f"| **{ai:.1f}** FLOP/B | {d_cap_str} | {d_sav_str} | {l_cap_str} | {l_sav_str} | {div_str} |\n")

        f.write("\n## 3. Scientific Analysis & Real-Time Governor Heuristics\n\n")
        f.write("1. **Early Saturation of L2-Resident Workloads:**\n")
        f.write("   Because L2 cache provides over $4.5\\times$ greater bandwidth than the GDDR6 bus, workloads hitting L2 cache hit their ALU compute ceiling at much lower arithmetic intensities ($I \\ge 25$ vs $I \\ge 113$).\n")
        f.write("   Consequently, the Golden Zone for L2-resident kernels rapidly shifts to the unconstrained baseline (250 W) well before DRAM-resident workloads.\n")
        f.write("2. **Governor Classification Rule:**\n")
        f.write("   When the CUPTI profiler samples operational intensity, the governor must cross-reference DRAM bandwidth vs L2 bandwidth (`dram__bytes.sum` vs `lts__t_bytes.sum`). If L2 hit rate $> 80\\%$, the governor must evaluate against $I^*_{\\text{L2}} \\approx 25\\text{ FLOP/B}$ instead of $I^*_{\\text{DRAM}} \\approx 113\\text{ FLOP/B}$ to avoid inducing massive performance stalls.\n")

    print(f"\nGenerated Comparative Summary Report: {report_path}")

def run_experiment():
    parser = argparse.ArgumentParser(description="Hierarchical Roofline AI Sweep (DRAM vs L2 Cache)")
    parser.add_argument("--memory-target", type=str, choices=["both", "dram", "l2"], default="both",
                        help="Memory hierarchy to benchmark: 'both' (default), 'dram', or 'l2'")
    parser.add_argument("--ai-min", type=float, default=0.0, help="Minimum Arithmetic Intensity")
    parser.add_argument("--ai-max", type=float, default=200.0, help="Maximum Arithmetic Intensity")
    parser.add_argument("--ai-step", type=float, default=10.0, help="Arithmetic Intensity step size")
    parser.add_argument("--power-caps", type=int, nargs="+", 
                        default=[250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100],
                        help="List of power caps in Watts to test (default: 250 down to 100 in steps of 10)")
    parser.add_argument("--target-duration-s", type=float, default=2.5,
                        help="Target execution duration per power cap trial in seconds (default: 2.5s for steady-state sampling)")
    parser.add_argument("--deg-threshold", type=float, default=5.0,
                        help="Maximum permissible runtime degradation %% for Golden Zone (default: 5.0%%)")
    parser.add_argument("--dram-buffer-mb", type=int, default=256,
                        help="Buffer size in MB for DRAM regime (default: 256 MB)")
    parser.add_argument("--l2-buffer-mb", type=int, default=16,
                        help="Buffer size in MB for L2 Cache regime (default: 16 MB, fits inside 64MB L2)")
    parser.add_argument("--dram-iterations", type=int, default=30,
                        help="Iterations for DRAM benchmark passes (default: 30)")
    parser.add_argument("--l2-iterations", type=int, default=400,
                        help="Iterations for L2 benchmark passes (default: 400 to normalize elapsed time)")
    parser.add_argument("--warmup-iters", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--saturation-count", type=int, default=3,
                        help="Stop if Golden Zone saturates at baseline for N consecutive steps (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode: runs without setting hardware power caps")
    parser.add_argument("--output-dir", type=str, default="experiment/results",
                        help="Directory to store results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.init()
    device = torch.device('cuda:0')
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device_name = pynvml.nvmlDeviceGetName(nvml_handle)

    default_limit_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(nvml_handle)
    default_limit_w = default_limit_mw / 1000.0

    print("=" * 80)
    print(f"HIERARCHICAL REAL-TIME ROOFLINE: DRAM vs. L2 CACHE GOLDEN ZONES")
    print(f"Device: {device_name}")
    print(f"Selected Memory Target: {args.memory_target.upper()}")
    print(f"AI Range: [{args.ai_min:.1f} -> {args.ai_max:.1f}] step {args.ai_step:.1f} FLOPs/Byte")
    print(f"Power Caps to Sweep: {args.power_caps} W")
    print(f"Degradation Tolerance: <= {args.deg_threshold:.1f}%")
    if args.dry_run:
        print("[MODE] DRY-RUN / TEST MODE: Hardware power capping is simulated.")
    print("=" * 80)

    if not args.dry_run:
        test_ok, msg = set_gpu_power_cap(nvml_handle, args.power_caps[0])
        if not test_ok:
            print(f"\n[WARNING] {msg}")
            print("[WARNING] Please run with sudo to enable hardware power capping:")
            print(f"          sudo /home/antpc/anaconda3/envs/kernel-bench/bin/python experiment/benchmark_ai_powercaps.py")
            print("[INFO] Or use --dry-run to test kernel execution without changing power limits.")
            sys.exit(1)
        else:
            print(f"[STATUS] Hardware Power Capping Verified via: {msg}\n")
    else:
        print("[STATUS] Running in dry-run mode.\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(script_dir, "intensity_benchmark_kernels.so")
    kernel_lib = load_kernel_library(so_path)
    stream = torch.cuda.current_stream().cuda_stream

    dram_gz = None
    l2_gz = None

    try:
        # Regime 1: GDDR DRAM
        if args.memory_target in ["both", "dram"]:
            _, dram_gz = run_single_regime(
                regime_name="DRAM",
                buffer_mb=args.dram_buffer_mb,
                iterations=args.dram_iterations,
                warmup_iters=args.warmup_iters,
                args=args,
                nvml_handle=nvml_handle,
                kernel_lib=kernel_lib,
                stream=stream,
                device=device
            )

        # Regime 2: L2 Cache
        if args.memory_target in ["both", "l2"]:
            _, l2_gz = run_single_regime(
                regime_name="L2",
                buffer_mb=args.l2_buffer_mb,
                iterations=args.l2_iterations,
                warmup_iters=args.warmup_iters * 4, # Warmup ensures 100% L2 residency
                args=args,
                nvml_handle=nvml_handle,
                kernel_lib=kernel_lib,
                stream=stream,
                device=device
            )

        # Comparative Report
        if args.memory_target == "both" and dram_gz is not None and l2_gz is not None:
            generate_comparative_report(dram_gz, l2_gz, args.output_dir, args.deg_threshold)

    finally:
        if not args.dry_run:
            print("\nRestoring GPU power cap to default...")
            reset_gpu_power_cap(nvml_handle, default_limit_w)
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    run_experiment()
