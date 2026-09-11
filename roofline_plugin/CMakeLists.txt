cmake_minimum_required(VERSION 3.18)
project(RooflinePlugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Paths for CUPTI and NVPERF
set(CUPTI_DIR "D:/NVIDIA CUDA Toolkit 12.9/extras/CUPTI")
set(CUPTI_INCLUDE_DIR "${CUPTI_DIR}/include")
set(CUPTI_LIB_DIR "${CUPTI_DIR}/lib64")

# We need the profilerhost_util files we copied
set(UTIL_SRC 
    Eval.cpp
    Metric.cpp
    List.cpp
)

add_library(roofline_plugin SHARED roofline_plugin.cpp ${UTIL_SRC})

target_include_directories(roofline_plugin PRIVATE 
    ${CUPTI_INCLUDE_DIR}
    ${CUPTI_DIR}/samples/common
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(roofline_plugin PRIVATE ${CUPTI_LIB_DIR})

target_link_libraries(roofline_plugin PRIVATE 
    CUDA::cuda_driver 
    CUDA::cudart 
    cupti 
    nvperf_host 
    nvperf_target
)

# Export the DLL correctly on Windows
target_compile_definitions(roofline_plugin PRIVATE -DDLL_EXPORT)
