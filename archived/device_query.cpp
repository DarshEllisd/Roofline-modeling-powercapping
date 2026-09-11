#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    
    int devId = 0; // Just use GPU 0 for now
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
    
    // Cores per SM lookup (simplified for modern GPUs)
    int cores_per_sm = 0;
    if (prop.major == 8 || prop.major == 9) cores_per_sm = 128; // Ampere, Ada, Hopper
    else if (prop.major == 7) cores_per_sm = 64; // Volta, Turing
    else cores_per_sm = 128; // Fallback
    
    // Calculate P_peak in TFLOP/s
    // clockRate is in kHz
    double max_clock_mhz = prop.clockRate / 1000.0;
    double p_peak_gflops = (prop.multiProcessorCount * cores_per_sm * max_clock_mhz * 2.0) / 1000.0;
    std::cout << "Max Core Clock: " << max_clock_mhz << " MHz" << std::endl;
    std::cout << "Calculated P_peak (FP32): " << p_peak_gflops / 1000.0 << " TFLOP/s" << std::endl;
    
    // Calculate B_peak in GB/s
    // memoryClockRate is in kHz, memoryBusWidth is in bits
    double mem_clock_mhz = prop.memoryClockRate / 1000.0;
    // For GDDR6/GDDR6X, data is transferred on multiple edges. 
    // Usually memoryClockRate * busWidth * 2 (for GDDR/DDR) / 8
    double b_peak_gbs = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) * 2.0) / 1e9;
    std::cout << "Memory Clock: " << mem_clock_mhz << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << "-bit" << std::endl;
    std::cout << "Calculated B_peak: " << b_peak_gbs << " GB/s" << std::endl;
    
    // Hardware Ridge Point (I*)
    // I* = (P_peak in GFLOPs) / (B_peak in GB/s) -> FLOPs/Byte
    double ridge_point = p_peak_gflops / b_peak_gbs;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Hardware Ridge Point (I*): " << ridge_point << " FLOPs/Byte" << std::endl;
    
    return 0;
}
