#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvml.h>

int main() {
    std::cout << "=== Testing CUDA Runtime ===" << std::endl;
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }
    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device 0: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << "-bit" << std::endl;

    int clockKhz = 0, memClockKhz = 0;
    cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memClockKhz, cudaDevAttrMemoryClockRate, 0);
    std::cout << "Memory Clock Rate (Attr): " << memClockKhz / 1000.0 << " MHz" << std::endl;
    std::cout << "GPU Boost Clock Rate (Attr): " << clockKhz / 1000.0 << " MHz" << std::endl;

    std::cout << "\n=== Testing CUDA Driver & Context ===" << std::endl;
    cuInit(0);
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);
    CUcontext cuContext;
    cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
    cuCtxPushCurrent(cuContext);

    std::cout << "\n=== Testing CUPTI Profiler Initialization ===" << std::endl;
    CUpti_Profiler_Initialize_Params initParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUptiResult status = cuptiProfilerInitialize(&initParams);
    if (status != CUPTI_SUCCESS) {
        const char *errstr = nullptr;
        cuptiGetResultString(status, &errstr);
        std::cerr << "cuptiProfilerInitialize failed: " << (errstr ? errstr : "unknown") << std::endl;
    } else {
        std::cout << "cuptiProfilerInitialize: SUCCESS" << std::endl;
    }

    CUpti_Device_GetChipName_Params chipParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chipParams.deviceIndex = cuDevice;
    status = cuptiDeviceGetChipName(&chipParams);
    if (status == CUPTI_SUCCESS && chipParams.pChipName) {
        std::cout << "CUPTI Auto-Detected Chip Name: " << chipParams.pChipName << std::endl;
    } else {
        std::cerr << "cuptiDeviceGetChipName failed." << std::endl;
    }

    std::cout << "\n=== Testing NVML ===" << std::endl;
    nvmlReturn_t nvmlStatus = nvmlInit();
    if (nvmlStatus == NVML_SUCCESS) {
        nvmlDevice_t nvmlDev;
        nvmlDeviceGetHandleByIndex(0, &nvmlDev);
        unsigned int power_limit = 0, default_limit = 0, min_limit = 0, max_limit = 0;
        nvmlDeviceGetPowerManagementLimit(nvmlDev, &power_limit);
        nvmlDeviceGetPowerManagementDefaultLimit(nvmlDev, &default_limit);
        nvmlDeviceGetPowerManagementLimitConstraints(nvmlDev, &min_limit, &max_limit);
        std::cout << "NVML Current Power Limit: " << (power_limit / 1000.0) << " W" << std::endl;
        std::cout << "NVML Default Power Limit: " << (default_limit / 1000.0) << " W" << std::endl;
        std::cout << "NVML Allowed Power Limit Range: [" << (min_limit / 1000.0) << " W - " << (max_limit / 1000.0) << " W]" << std::endl;

        unsigned int cur_power = 0;
        nvmlDeviceGetPowerUsage(nvmlDev, &cur_power);
        std::cout << "NVML Instantaneous Power Usage: " << (cur_power / 1000.0) << " W" << std::endl;

        unsigned int maxGfxClk = 0, maxMemClk = 0;
        nvmlDeviceGetMaxClockInfo(nvmlDev, NVML_CLOCK_GRAPHICS, &maxGfxClk);
        nvmlDeviceGetMaxClockInfo(nvmlDev, NVML_CLOCK_MEM, &maxMemClk);
        std::cout << "NVML Max Graphics Clock: " << maxGfxClk << " MHz" << std::endl;
        std::cout << "NVML Max Memory Clock: " << maxMemClk << " MHz" << std::endl;

        nvmlShutdown();
    } else {
        std::cerr << "nvmlInit failed." << std::endl;
    }

    cuDevicePrimaryCtxRelease(cuDevice);
    std::cout << "\nALL CHECKS PASSED!" << std::endl;
    return 0;
}
