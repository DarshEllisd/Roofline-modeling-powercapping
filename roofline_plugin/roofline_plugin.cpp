//
// Roofline Plugin using CUPTI Profiling API & NVML Power Capping
// Hardware: NVIDIA RTX 5000 Ada Generation (Workstation)
// Power Capping Range: [100W, 250W] | Baseline TDP: 250W
// SLA Target: Runtime degradation <= 8.0%
//

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <mutex>
#include <iomanip>
#include <cmath>
#include <atomic>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvml.h>

#include "Eval.h"
#include "Metric.h"
#include "golden_zone_heuristics.hpp"

#if defined(_WIN32) || defined(_WIN64)
  #define PLUGIN_API __declspec(dllexport)
#else
  #define PLUGIN_API __attribute__((visibility("default")))
#endif

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        std::cerr << "[CUPTI ERROR] " << #apiFuncCall << " failed with " << errstr << std::endl; \
    }                                                                          \
} while(0)

// The metrics needed for Hierarchical Roofline Analysis
static std::vector<std::string> metricNames = {
    "dram__bytes.sum", 
    "lts__t_bytes.sum",
    "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum"
};

// Global variables for the background profiling thread
static bool g_keepRunning = true;
static std::thread g_profilerThread;
static const size_t s_MaxRanges = 16;

// Golden Zone Governor State (Power Capping Mode - No Clock Locking)
static std::atomic<bool> g_governorEnabled(false);
static std::atomic<uint32_t> g_currentPowerCapW(250);
static std::atomic<uint32_t> g_defaultPowerCapW(250);
static std::atomic<int> g_currentState(0); // 0: IDLE, 1: DRAM-BOUND, 2: L2-BOUND, 3: COMPUTE-BOUND
static std::chrono::steady_clock::time_point g_lastSwitchTime;
static const std::chrono::milliseconds g_dwellTime(1014); // T_dwell = 1000ms + 14ms

static nvmlDevice_t g_nvmlDevice = nullptr;
static bool g_nvmlInitialized = false;

static bool init_nvml() {
    if (g_nvmlInitialized) return true;
    if (nvmlInit() == NVML_SUCCESS) {
        if (nvmlDeviceGetHandleByIndex(0, &g_nvmlDevice) == NVML_SUCCESS) {
            unsigned int default_limit_mw = 250000;
            if (nvmlDeviceGetPowerManagementDefaultLimit(g_nvmlDevice, &default_limit_mw) == NVML_SUCCESS) {
                g_defaultPowerCapW = default_limit_mw / 1000;
            }
            unsigned int cur_limit_mw = 250000;
            if (nvmlDeviceGetPowerManagementLimit(g_nvmlDevice, &cur_limit_mw) == NVML_SUCCESS) {
                g_currentPowerCapW = cur_limit_mw / 1000;
            }
            g_nvmlInitialized = true;
            return true;
        }
    }
    return false;
}

// Sets the dynamic power limit in Watts via NVML (No GPU clocks modified)
static bool apply_gpu_power_cap(uint32_t cap_w) {
    if (!init_nvml()) return false;
    if (cap_w < 100) cap_w = 100;
    if (cap_w > 250) cap_w = 250;
    
    unsigned int power_mw = cap_w * 1000;
    nvmlReturn_t status = nvmlDeviceSetPowerManagementLimit(g_nvmlDevice, power_mw);
    if (status == NVML_SUCCESS) {
        g_currentPowerCapW = cap_w;
        return true;
    } else {
        const char* errStr = nvmlErrorString(status);
        std::cerr << "[RooflinePlugin] nvmlDeviceSetPowerManagementLimit(" << cap_w 
                  << "W) failed: " << errStr << " (requires CAP_SYS_ADMIN/root privileges)" << std::endl;
        return false;
    }
}

static void reset_gpu_power_cap() {
    if (!init_nvml()) return;
    uint32_t default_cap = g_defaultPowerCapW.load();
    if (default_cap == 0) default_cap = 250;
    apply_gpu_power_cap(default_cap);
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

extern "C" PLUGIN_API void start_profiling(double p_peak_tflops, double b_peak_gbs);

extern "C" PLUGIN_API void start_profiling_auto() {
    start_profiling(0.0, 0.0);
}

extern "C" PLUGIN_API void start_profiling(double p_peak_tflops, double b_peak_gbs) {
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
        std::string chipName = (getChipNameParams.pChipName != nullptr) ? getChipNameParams.pChipName : "AD102";
        std::cout << "[RooflinePlugin] Auto-Detected GPU Chip: " << chipName << std::endl;

        // Auto-calculate P_peak and B_peak if not manually specified
        int numSMs = 100;
        int clockRateKHz = 0;
        int memClockKHz = 0;
        int busWidth = 256;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cuDevice);
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, cuDevice);
        cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, cuDevice);
        cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, cuDevice);

        if (p_peak_tflops <= 0.0) {
            double maxClockHz = (clockRateKHz > 0) ? (clockRateKHz * 1000.0) : (2550.0 * 1e6);
            int coresPerSM = 128;
            p_peak_tflops = (numSMs * coresPerSM * maxClockHz * 2.0) / 1e12;
            std::cout << "[RooflinePlugin] Peak FP32 Compute: " << std::fixed << std::setprecision(2) << p_peak_tflops << " TFLOP/s (" << numSMs << " SMs)" << std::endl;
        }

        if (b_peak_gbs <= 0.0) {
            if (memClockKHz > 0) {
                double memClockHz = memClockKHz * 1000.0;
                b_peak_gbs = ((busWidth / 8.0) * memClockHz * 2.0) / 1e9;
            } else {
                b_peak_gbs = 576.0; // RTX 5000 Ada GDDR6 default
            }
            std::cout << "[RooflinePlugin] Peak GDDR DRAM Bandwidth: " << std::fixed << std::setprecision(2) << b_peak_gbs << " GB/s" << std::endl;
        }

        double ridge_point_dram = (p_peak_tflops * 1000.0) / b_peak_gbs;
        std::cout << "[RooflinePlugin] Empirical DRAM Ridge Point: " << std::fixed << std::setprecision(2) << ridge_point_dram << " FLOPs/Byte" << std::endl;
        std::cout << "[RooflinePlugin] Empirical L2 Ridge Point: ~16.75 FLOPs/Byte" << std::endl;

        // 3. Generate ConfigImage
        std::vector<uint8_t> configImage;
        if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage)) {
            std::cerr << "[RooflinePlugin] Failed to generate ConfigImage for " << chipName << std::endl;
            return;
        }
        
        // 4. Generate CounterDataPrefix
        std::vector<uint8_t> counterDataImagePrefix;
        if (!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix)) {
            std::cerr << "[RooflinePlugin] Failed to generate CounterDataPrefixImage" << std::endl;
            return;
        }
        
        // 5. Create CounterDataImage
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
            
            // Sample for 1000ms (1 second window)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
            CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
            
            // Evaluate Metrics and measure decode latency
            auto t_start = std::chrono::high_resolution_clock::now();
            std::vector<NV::Metric::Eval::MetricNameValue> metricNameValueMap;
            NV::Metric::Eval::GetMetricGpuValue(chipName, counterDataImage, metricNames, metricNameValueMap);
            auto t_end = std::chrono::high_resolution_clock::now();
            double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            double dram_bytes = 0.0;
            double l2_bytes = 0.0;
            double flops = 0.0;
            
            for (const auto& metric : metricNameValueMap) {
                for (const auto& rangeVal : metric.rangeNameMetricValueMap) {
                    if (std::isnan(rangeVal.second) || rangeVal.second < 0.0) continue;
                    if (metric.metricName == "dram__bytes.sum") {
                        dram_bytes += rangeVal.second;
                    } else if (metric.metricName == "lts__t_bytes.sum") {
                        l2_bytes += rangeVal.second;
                    } else if (metric.metricName == "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum") {
                        flops += rangeVal.second;
                    }
                }
            }
            
            const double IDLE_BYTES_THRESHOLD = 10.0 * 1024.0 * 1024.0; // 10 MB noise floor
            bool is_idle = (flops == 0.0 && dram_bytes < IDLE_BYTES_THRESHOLD && l2_bytes < IDLE_BYTES_THRESHOLD);
            
            bool is_l2_resident = false;
            double intensity = 0.0;
            int state = 0; // 0: IDLE, 1: DRAM-BOUND, 2: L2-BOUND, 3: COMPUTE-BOUND
            
            if (is_idle) {
                state = 0;
                intensity = 0.0;
            } else {
                // Classify memory hierarchy residency (L2 vs DRAM)
                if (l2_bytes > 2.0 * dram_bytes || (dram_bytes < IDLE_BYTES_THRESHOLD && l2_bytes >= IDLE_BYTES_THRESHOLD)) {
                    is_l2_resident = true;
                    intensity = (l2_bytes > 0.0) ? (flops / l2_bytes) : 0.0;
                    if (intensity > 16.75) {
                        state = 3; // COMPUTE-BOUND
                    } else {
                        state = 2; // L2-BOUND
                    }
                } else {
                    is_l2_resident = false;
                    intensity = (dram_bytes > 0.0) ? (flops / dram_bytes) : 0.0;
                    if (intensity > ridge_point_dram) {
                        state = 3; // COMPUTE-BOUND
                    } else {
                        state = 1; // DRAM-BOUND
                    }
                }
            }
            g_currentState = state;

            // Output real-time CUPTI telemetry
            std::cout << "[CUPTI] FLOPs: " << std::setw(12) << (uint64_t)flops 
                      << " | DRAM: " << std::setw(5) << (uint64_t)(dram_bytes / (1024*1024)) << " MB"
                      << " | L2: " << std::setw(5) << (uint64_t)(l2_bytes / (1024*1024)) << " MB"
                      << " | Regime: " << (is_l2_resident ? "L2  " : "DRAM")
                      << " | AI: " << std::fixed << std::setprecision(2) << std::setw(6) << intensity
                      << " -> ";
            
            if (state == 0) std::cout << "[IDLE         ]";
            else if (state == 1) std::cout << "[DRAM-BOUND   ]";
            else if (state == 2) std::cout << "[L2-BOUND     ]";
            else if (state == 3) std::cout << "[COMPUTE-BOUND]";

            std::cout << " (Decode: " << std::fixed << std::setprecision(1) << decode_ms << "ms)";

            // Golden Zone Power Capping Governor Evaluation (under <= 8.0% SLA heuristics)
            if (g_governorEnabled.load()) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_lastSwitchTime);
                
                uint32_t target_cap_w = 250;
                if (state == 0) {
                    target_cap_w = g_defaultPowerCapW.load(); // Baseline TDP on idle
                } else {
                    // Fast O(1) heuristic lookup from golden_zone_heuristics.hpp
                    target_cap_w = get_golden_zone_power_cap_8pct(intensity, is_l2_resident);
                }

                if (g_currentPowerCapW.load() != target_cap_w) {
                    if (elapsed >= g_dwellTime) {
                        if (apply_gpu_power_cap(target_cap_w)) {
                            g_lastSwitchTime = now;
                            std::cout << " -> [GOVERNOR: POWER CAP " << target_cap_w << "W ("
                                      << (is_l2_resident ? "L2" : "DRAM") << " AI=" 
                                      << std::fixed << std::setprecision(1) << intensity << ")]";
                        }
                    } else {
                        std::cout << " -> [GOVERNOR: HOLDING " << g_currentPowerCapW.load() 
                                  << "W (Dwell: " << elapsed.count() << "ms, Next: " << target_cap_w << "W)]";
                    }
                } else {
                    std::cout << " -> [GOVERNOR: OPTIMAL (" << target_cap_w << "W)]";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        cuCtxPopCurrent(&cuContext);
    });
    
    g_profilerThread.detach();
}

extern "C" PLUGIN_API void enable_governor() {
    g_governorEnabled = true;
    g_lastSwitchTime = std::chrono::steady_clock::now() - g_dwellTime; // Allow immediate first switch
    init_nvml();
    std::cout << "[RooflinePlugin] Golden Zone Power Capping Governor ENABLED (SLA: <=8.0% | Dynamic Caps: 100W-250W | Dwell: " 
              << g_dwellTime.count() << "ms)" << std::endl;
}

// Legacy signature support (ignores clock arguments and logs explanation)
extern "C" PLUGIN_API void enable_governor_legacy(uint32_t mem_clock_mhz, uint32_t comp_clock_mhz) {
    std::cout << "[RooflinePlugin] Note: Operating in Workstation Power Capping Mode. GPU clocks are controlled natively by GPU Boost." << std::endl;
    enable_governor();
}

extern "C" PLUGIN_API void disable_governor() {
    g_governorEnabled = false;
    if (g_nvmlInitialized) {
        reset_gpu_power_cap();
    }
    std::cout << "[RooflinePlugin] Golden Zone Governor DISABLED (Restored default power cap: " 
              << g_defaultPowerCapW.load() << "W)." << std::endl;
}

extern "C" PLUGIN_API int get_profiler_state() {
    return g_currentState.load();
}

extern "C" PLUGIN_API uint32_t get_active_power_cap() {
    return g_currentPowerCapW.load();
}

// Retained for legacy ABI compatibility
extern "C" PLUGIN_API uint32_t get_active_clock() {
    return 0; // GPU clocks are not locked on workstation
}

extern "C" PLUGIN_API void set_power_cap_manual(uint32_t cap_w) {
    apply_gpu_power_cap(cap_w);
}

extern "C" PLUGIN_API void stop_profiling() {
    g_keepRunning = false;
    if (g_nvmlInitialized) {
        reset_gpu_power_cap();
        nvmlShutdown();
        g_nvmlInitialized = false;
        std::cout << "[RooflinePlugin] GPU Power Cap restored to " 
                  << g_defaultPowerCapW.load() << "W & NVML closed." << std::endl;
    }
}
