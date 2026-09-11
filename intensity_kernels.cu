#include <cuda_runtime.h>
#include <stdio.h>

// Template kernel for exact compile-time unrolling of FMA operations
template<int N_FMA>
__global__ void intensity_kernel_unrolled(float* __restrict__ out, const float* __restrict__ in, int N, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];
        #pragma unroll
        for (int i = 0; i < N_FMA; ++i) {
            x = fmaf(x, 1.00001f, val);
        }
        out[idx] = x;
    }
}

// Dynamic loop kernel where N_FMA can be specified at runtime
__global__ void intensity_kernel_dynamic(float* __restrict__ out, const float* __restrict__ in, int N, int n_fma, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];
        for (int i = 0; i < n_fma; ++i) {
            x = fmaf(x, 1.00001f, val);
        }
        out[idx] = x;
    }
}

extern "C" {

// Symmetric Linear Pair: Delta = 30 FLOPs/Byte from Ridge Point (56.79)
// Memory: 107 FMAs = 214 FLOPs / 8 Bytes = 26.75 FLOPs/Byte (56.79 - 30.04)
__declspec(dllexport) void launch_symmetric_memory_kernel(float* d_out, const float* d_in, int N, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    intensity_kernel_unrolled<107><<<numBlocks, blockSize, 0, stream>>>(d_out, d_in, N, 0.00001f);
}

// Exactly at Ridge Point: 227 FMAs = 454 FLOPs / 8 Bytes = 56.75 FLOPs/Byte (I* = 56.79)
__declspec(dllexport) void launch_ridge_point_kernel(float* d_out, const float* d_in, int N, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    intensity_kernel_unrolled<227><<<numBlocks, blockSize, 0, stream>>>(d_out, d_in, N, 0.00001f);
}

// Compute: 347 FMAs = 694 FLOPs / 8 Bytes = 86.75 FLOPs/Byte (56.79 + 29.96)
__declspec(dllexport) void launch_symmetric_compute_kernel(float* d_out, const float* d_in, int N, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    intensity_kernel_unrolled<347><<<numBlocks, blockSize, 0, stream>>>(d_out, d_in, N, 0.00001f);
}

// General dynamic launcher
__declspec(dllexport) void launch_custom_intensity_kernel(float* d_out, const float* d_in, int N, int n_fma, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    intensity_kernel_dynamic<<<numBlocks, blockSize, 0, stream>>>(d_out, d_in, N, n_fma, 0.00001f);
}

}
