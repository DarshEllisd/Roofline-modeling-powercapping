import time
import torch
import ctypes
import os

print("Initializing PyTorch CUDA Context...")
torch.cuda.init()
_ = torch.tensor([1.0], device='cuda') # Force PyTorch to create the CUDA context
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Add CUDA CUPTI DLLs to search path
os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\extras\CUPTI\lib64")
os.add_dll_directory(r"D:\NVIDIA CUDA Toolkit 12.9\bin")

# Load the DLL
dll_path = os.path.join(os.path.dirname(__file__), 'roofline_plugin.dll')
print(f"Loading DLL: {dll_path}")
plugin = ctypes.CDLL(dll_path)

# Start profiling in background thread with 100% automated GPU discovery
print("\n>>> STARTING TRUE CUPTI ROOFLINE PROFILER (AUTO-DISCOVERY) <<<")
plugin.start_profiling_auto()

time.sleep(1.0)

print("\n--- PHASE 1: Running MEMORY-BOUND Workload (Element-wise Tanh/GELU on 16M floats) ---")
mem_tensor = torch.randn(16 * 1024 * 1024, device='cuda')
start_time = time.time()
while time.time() - start_time < 5.0:
    mem_tensor = torch.tanh(mem_tensor) + torch.sigmoid(mem_tensor)
    torch.cuda.synchronize()
    time.sleep(0.05)

print("\n--- PHASE 2: Running COMPUTE-BOUND Workload (Heavy 4096x4096 GEMM) ---")
mat_a = torch.randn(4096, 4096, device='cuda')
mat_b = torch.randn(4096, 4096, device='cuda')
start_time = time.time()
while time.time() - start_time < 5.0:
    mat_c = torch.matmul(mat_a, mat_b)
    torch.cuda.synchronize()
    time.sleep(0.05)

print("\n--- PHASE 3: IDLE ---")
time.sleep(2.0)

print("\nStopping Profiler...")
plugin.stop_profiling()
time.sleep(1.0)
print("Test Complete.")
