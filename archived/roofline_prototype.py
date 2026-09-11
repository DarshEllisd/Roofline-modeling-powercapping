import time
import sys
import pynvml
import argparse

# --- RTX 4050 Laptop Specific Constants ---
# Ada Lovelace (Compute Capability 8.9) has 128 FP32 cores per SM
CORES_PER_SM = 128

def main(interval):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    
    # 1. Query Hardware Properties
    # SM Count (Note: NVML doesn't expose SM count directly, we hardcode it for RTX 4050 Laptop)
    # A robust C++ daemon would use cudaGetDeviceProperties, but we know it's 20 for this GPU.
    sm_count = 20 
    
    # Max Clocks
    max_gpu_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    max_mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    
    # Calculate P_peak (Peak Compute in GFLOPs)
    # P_peak = SMs * Cores/SM * MaxClock * 2 (FMA)
    p_peak_gflops = (sm_count * CORES_PER_SM * max_gpu_clock_mhz * 2) / 1000.0
    
    # Calculate B_peak (Peak Memory Bandwidth in GB/s)
    # RTX 4050 Laptop has a 96-bit memory bus
    mem_bus_width = 96
    # GDDR6 transfers data on dual edges, memory clock in NVML is already 'effective' or base depending on driver.
    # Usually: (Effective Clock MHz * Bus Width) / 8. 
    # For RTX 4050 mobile, max bandwidth is usually around 192 GB/s. Let's hardcode the bus width to calculate it.
    b_peak_gbs = (max_mem_clock_mhz * mem_bus_width * 2) / (8 * 1000.0) 
    
    # Ridge Point
    ridge_point = p_peak_gflops / b_peak_gbs

    print("=" * 60)
    print(f"Device: {name}")
    print(f"Peak Compute (P_peak):  {p_peak_gflops/1000.0:.2f} TFLOP/s")
    print(f"Peak Mem Bw (B_peak):   {b_peak_gbs:.2f} GB/s")
    print(f"Hardware Ridge Point:   {ridge_point:.2f} FLOPs/Byte")
    print("=" * 60)
    print("Starting Pseudo-Roofline Live Classification...\n")

    try:
        while True:
            # Poll NVML Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            mem_util = util.memory
            
            # Approximate Achieved Metrics based on Utilization
            # This is a 'Pseudo' Roofline. A real one uses CUPTI bytes/flops.
            achieved_gflops = p_peak_gflops * (gpu_util / 100.0)
            achieved_gbs = b_peak_gbs * (mem_util / 100.0)
            
            # Prevent division by zero
            safe_gbs = max(achieved_gbs, 1.0)
            
            # Live Operational Intensity
            intensity = achieved_gflops / safe_gbs
            
            # Classify
            if gpu_util < 5 and mem_util < 5:
                regime = "IDLE"
            elif intensity < ridge_point:
                regime = "MEMORY-BOUND"
            else:
                regime = "COMPUTE-BOUND"
                
            print(f"[{regime:<13}] Intensity: {intensity:6.1f} | Ridge: {ridge_point:.1f} | GPU Util: {gpu_util:3d}% | Mem Util: {mem_util:3d}%")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Pseudo-Roofline Classifier")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    args = parser.parse_args()
    main(args.interval)
