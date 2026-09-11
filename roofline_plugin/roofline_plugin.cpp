//
// Roofline Plugin DLL using CUPTI Profiling API
//

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <mutex>
#include <iomanip>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvml.h>
#include <atomic>

#include "Eval.h"
#include "Metric.h"

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        std::cerr << "[CUPTI ERROR] " << #apiFuncCall << " failed with " << errstr << std::endl; \
    }                                                                          \
} while(0)

// The metrics we need for the Roofline
static std::vector<std::string> metricNames = {
    "dram__bytes.sum", 
    "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum"
};

// Global variables for the background thread
static bool g_keepRunning = true;
static std::thread g_profilerThread;
static const size_t s_MaxRanges = 16;

// Golden Zone Governor State
static std::atomic<bool> g_governorEnabled(false);
static std::atomic<uint32_t> g_memGoldenClock(945);
static std::atomic<uint32_t> g_compGoldenClock(1950);
static std::atomic<uint32_t> g_currentLockedClock(0);
static std::atomic<int> g_currentState(0); // 0: IDLE, 1: MEMORY, 2: COMPUTE
static std::chrono::steady_clock::time_point g_lastSwitchTime;
static const std::chrono::milliseconds g_dwellTime(1014); // T_dwell = 1000ms + 14ms

static nvmlDevice_t g_nvmlDevice = nullptr;
static bool g_nvmlInitialized = false;

static bool init_nvml() {
    if (g_nvmlInitialized) return true;
    if (nvmlInit() == NVML_SUCCESS) {
        if (nvmlDeviceGetHandleByIndex(0, &g_nvmlDevice) == NVML_SUCCESS) {
            g_nvmlInitialized = true;
            return true;
        }
    }
    return false;
}

static void apply_gpu_clock(uint32_t clock_mhz) {
    if (!init_nvml()) return;
    if (clock_mhz == 0) {
        nvmlDeviceResetGpuLockedClocks(g_nvmlDevice);
        g_currentLockedClock = 0;
    } else {
        if (nvmlDeviceSetGpuLockedClocks(g_nvmlDevice, clock_mhz, clock_mhz) == NVML_SUCCESS) {
            g_currentLockedClock = clock_mhz;
        }
    }
}

bool CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = s_MaxRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = s_MaxRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}

extern "C" __declspec(dllexport) void start_profiling(double p_peak_tflops, double b_peak_gbs);

extern "C" __declspec(dllexport) void start_profiling_auto() {
    start_profiling(0.0, 0.0);
}

extern "C" __declspec(dllexport) void start_profiling(double p_peak_tflops, double b_peak_gbs) {
    CUcontext cuContext;
    if (cuCtxGetCurrent(&cuContext) != CUDA_SUCCESS || cuContext == nullptr) {
        std::cerr << "[RooflinePlugin] Error: No active CUDA context found." << std::endl;
        return;
    }

    CUdevice cuDevice;
    cuCtxGetDevice(&cuDevice);

    g_profilerThread = std::thread([cuContext, cuDevice, p_peak_tflops, b_peak_gbs]() mutable {
        std::cout << "[RooflinePlugin] Profiling Thread Started." << std::endl;
        cuCtxPushCurrent(cuContext);

        // 1. Initialize CUPTI Profiler FIRST
        CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        // 2. Auto-detect GPU Chip Name dynamically (AD102, AD104, AD107, GA102, etc.)
        CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
        getChipNameParams.deviceIndex = cuDevice;
        CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
        std::string chipName = (getChipNameParams.pChipName != nullptr) ? getChipNameParams.pChipName : "AD107";
        std::cout << "[RooflinePlugin] Auto-Detected GPU Chip: " << chipName << std::endl;

        // Auto-calculate P_peak and B_peak if not manually specified
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cuDevice);

        if (p_peak_tflops <= 0.0) {
            int numSMs = prop.multiProcessorCount;
            double maxClockHz = prop.clockRate * 1000.0;
            // 128 FP32 cores per SM for Turing (7.5), Ampere (8.0/8.6), Ada (8.9), Hopper (9.0)
            int coresPerSM = 128;
            if (prop.major == 7 && prop.minor == 0) coresPerSM = 64; // Volta
            p_peak_tflops = (numSMs * coresPerSM * maxClockHz * 2.0) / 1e12;
            std::cout << "[RooflinePlugin] Auto-Calculated P_peak: " << std::fixed << std::setprecision(2) << p_peak_tflops << " TFLOP/s" << std::endl;
        }

        if (b_peak_gbs <= 0.0) {
            double memClockHz = prop.memoryClockRate * 1000.0;
            b_peak_gbs = ((prop.memoryBusWidth / 8.0) * memClockHz * 2.0) / 1e9;
            std::cout << "[RooflinePlugin] Auto-Calculated B_peak: " << std::fixed << std::setprecision(2) << b_peak_gbs << " GB/s" << std::endl;
        }

        double ridge_point = (p_peak_tflops * 1000.0) / b_peak_gbs;
        std::cout << "[RooflinePlugin] Hardware Ridge Point: " << std::fixed << std::setprecision(2) << ridge_point << " FLOPs/Byte" << std::endl;

        // 3. Generate ConfigImage
        std::vector<uint8_t> configImage;
        if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage)) {
            std::cerr << "[RooflinePlugin] Failed to generate ConfigImage for " << chipName << std::endl;
            return;
        }
        
        // 3. Generate CounterDataPrefix
        std::vector<uint8_t> counterDataImagePrefix;
        if (!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix)) {
            std::cerr << "[RooflinePlugin] Failed to generate CounterDataPrefixImage" << std::endl;
            return;
        }
        
        // 4. Create CounterDataImage
        std::vector<uint8_t> counterDataImage;
        std::vector<uint8_t> counterDataScratchBuffer;
        CreateCounterDataImage(counterDataImage, counterDataScratchBuffer, counterDataImagePrefix);
        
        CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
        beginSessionParams.ctx = cuContext;
        beginSessionParams.counterDataImageSize = counterDataImage.size();
        beginSessionParams.pCounterDataImage = &counterDataImage[0];
        beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
        beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
        beginSessionParams.replayMode = CUPTI_KernelReplay;
        beginSessionParams.range = CUPTI_AutoRange;
        beginSessionParams.maxRangesPerPass = s_MaxRanges;
        beginSessionParams.maxLaunchesPerPass = s_MaxRanges;
        
        CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
        setConfigParams.pConfig = &configImage[0];
        setConfigParams.configSize = configImage.size();
        setConfigParams.passIndex = 0;
        setConfigParams.minNestingLevel = 1;
        setConfigParams.numNestingLevels = 1;
        
        CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
        CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
        CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
        CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
        
        while (g_keepRunning) {
            // Reset counter image for new window
            CUpti_Profiler_CounterDataImage_Initialize_Params initParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
            CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
            counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
            counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
            counterDataImageOptions.maxNumRanges = s_MaxRanges;
            counterDataImageOptions.maxNumRangeTreeNodes = s_MaxRanges;
            counterDataImageOptions.maxRangeNameLength = 64;
            initParams.pOptions = &counterDataImageOptions;
            initParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
            initParams.counterDataImageSize = counterDataImage.size();
            initParams.pCounterDataImage = &counterDataImage[0];
            cuptiProfilerCounterDataImageInitialize(&initParams);

            CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
            CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
            CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
            
            // Sample for 1000ms (1 second)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
            CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
            
            // Evaluate Metrics and measure decode time
            auto t_start = std::chrono::high_resolution_clock::now();
            std::vector<NV::Metric::Eval::MetricNameValue> metricNameValueMap;
            NV::Metric::Eval::GetMetricGpuValue(chipName, counterDataImage, metricNames, metricNameValueMap);
            auto t_end = std::chrono::high_resolution_clock::now();
            double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            double dram_bytes = 0.0;
            double flops = 0.0;
            
            for (const auto& metric : metricNameValueMap) {
                for (const auto& rangeVal : metric.rangeNameMetricValueMap) {
                    if (std::isnan(rangeVal.second) || rangeVal.second < 0.0) continue;
                    if (metric.metricName == "dram__bytes.sum") {
                        dram_bytes += rangeVal.second;
                    }
                    if (metric.metricName == "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum") {
                        flops += rangeVal.second;
                    }
                }
            }
            
            double intensity = (dram_bytes > 0.0) ? (flops / dram_bytes) : 0.0;
            
            std::cout << "[CUPTI] FLOPs: " << std::setw(12) << (uint64_t)flops 
                      << " | Bytes: " << std::setw(12) << (uint64_t)dram_bytes 
                      << " | Intensity: " << std::fixed << std::setprecision(2) << std::setw(7) << intensity
                      << " -> ";
                      
            // True IDLE: Both compute and memory traffic are virtually zero (< 10 MB noise floor)
            const double IDLE_BYTES_THRESHOLD = 10.0 * 1024.0 * 1024.0;
            int state = 0; // 0: IDLE, 1: MEMORY, 2: COMPUTE
            if (flops == 0.0 && dram_bytes < IDLE_BYTES_THRESHOLD) {
                state = 0;
                std::cout << "[IDLE         ]";
            } else {
                if (intensity > ridge_point) {
                    state = 2;
                    std::cout << "[COMPUTE-BOUND]";
                } else {
                    state = 1;
                    std::cout << "[MEMORY-BOUND ]";
                }
            }
            g_currentState = state;
            std::cout << " (Decode: " << std::fixed << std::setprecision(1) << decode_ms << "ms)";

            // Golden Zone Governor Evaluation
            if (g_governorEnabled.load()) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_lastSwitchTime);

                if (state == 1) { // MEMORY-BOUND
                    uint32_t target = g_memGoldenClock.load();
                    if (g_currentLockedClock.load() != target) {
                        if (elapsed >= g_dwellTime) {
                            apply_gpu_clock(target);
                            g_lastSwitchTime = now;
                            std::cout << " -> [GOVERNOR: LOCKED " << target << " MHz (MEM GOLDEN ZONE)]";
                        } else {
                            std::cout << " -> [GOVERNOR: HOLDING (Dwell: " << elapsed.count() << "ms)]";
                        }
                    } else {
                        std::cout << " -> [GOVERNOR: OPTIMAL (" << target << " MHz)]";
                    }
                } else if (state == 2) { // COMPUTE-BOUND
                    uint32_t target = g_compGoldenClock.load();
                    if (g_currentLockedClock.load() != target) {
                        if (elapsed >= g_dwellTime) {
                            apply_gpu_clock(target);
                            g_lastSwitchTime = now;
                            std::cout << " -> [GOVERNOR: LOCKED " << target << " MHz (COMPUTE GOLDEN ZONE)]";
                        } else {
                            std::cout << " -> [GOVERNOR: HOLDING (Dwell: " << elapsed.count() << "ms)]";
                        }
                    } else {
                        std::cout << " -> [GOVERNOR: OPTIMAL (" << target << " MHz)]";
                    }
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        cuCtxPopCurrent(&cuContext);
    });
    
    g_profilerThread.detach();
}

extern "C" __declspec(dllexport) void enable_governor(uint32_t mem_clock_mhz, uint32_t comp_clock_mhz) {
    g_memGoldenClock = mem_clock_mhz;
    g_compGoldenClock = comp_clock_mhz;
    g_governorEnabled = true;
    g_lastSwitchTime = std::chrono::steady_clock::now() - g_dwellTime; // Allow immediate first switch
    init_nvml();
    std::cout << "[RooflinePlugin] Golden Zone Governor ENABLED (Memory Target: " 
              << mem_clock_mhz << " MHz | Compute Target: " << comp_clock_mhz << " MHz | Dwell: " 
              << g_dwellTime.count() << "ms)" << std::endl;
}

extern "C" __declspec(dllexport) void disable_governor() {
    g_governorEnabled = false;
    if (g_nvmlInitialized) {
        nvmlDeviceResetGpuLockedClocks(g_nvmlDevice);
        g_currentLockedClock = 0;
    }
    std::cout << "[RooflinePlugin] Golden Zone Governor DISABLED." << std::endl;
}

extern "C" __declspec(dllexport) int get_profiler_state() {
    return g_currentState.load();
}

extern "C" __declspec(dllexport) uint32_t get_active_clock() {
    return g_currentLockedClock.load();
}

extern "C" __declspec(dllexport) void stop_profiling() {
    g_keepRunning = false;
    if (g_nvmlInitialized) {
        nvmlDeviceResetGpuLockedClocks(g_nvmlDevice);
        g_currentLockedClock = 0;
        nvmlShutdown();
        g_nvmlInitialized = false;
        std::cout << "[RooflinePlugin] GPU Clocks reset to driver defaults & NVML closed." << std::endl;
    }
}
