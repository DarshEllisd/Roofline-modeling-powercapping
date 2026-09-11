#include <cuda_runtime.h>
#include <stdio.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// Kernel with loop unrolling for n_fma
__global__ void intensity_benchmark_kernel(float* __restrict__ out, const float* __restrict__ in, int N, int n_fma, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float x = in[i];

        // 16-way unrolled FMA block for maximum compiler instruction-level parallelism
        int full_blocks = n_fma / 16;
        int remainder = n_fma % 16;

        for (int b = 0; b < full_blocks; ++b) {
            #pragma unroll 16
            for (int k = 0; k < 16; ++k) {
                x = fmaf(x, 1.000001f, val);
            }
        }

        #pragma unroll 16
        for (int k = 0; k < 16; ++k) {
            if (k < remainder) {
                x = fmaf(x, 1.000001f, val);
            }
        }

        out[i] = x;
    }
}

extern "C" {

EXPORT_API void launch_intensity_benchmark(float* d_out, const float* d_in, int N, int n_fma, int iterations, cudaStream_t stream) {
    int blockSize = 256;
    // Launch enough blocks to saturate all 100 SMs of RTX 5000 Ada (e.g. 4 waves per SM = 800 blocks)
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 1600) numBlocks = 1600;

    for (int iter = 0; iter < iterations; ++iter) {
        intensity_benchmark_kernel<<<numBlocks, blockSize, 0, stream>>>(d_out, d_in, N, n_fma, 0.00001f);
    }
}

EXPORT_API int get_suggested_fma_count(double target_intensity) {
    // DRAM traffic per element = 4 bytes read + 4 bytes write = 8 bytes
    // FLOPs per element = 2 * N_FMA
    // Intensity = (2 * N_FMA) / 8 = N_FMA / 4
    // N_FMA = round(4 * target_intensity)
    int fma = (int)(target_intensity * 4.0 + 0.5);
    return (fma < 0) ? 0 : fma;
}

}
