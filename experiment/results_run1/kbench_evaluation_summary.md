# Real KBENCH Workloads Roofline Governor Evaluation

**Hardware:** NVIDIA RTX 5000 Ada Generation (100 SMs, 32GB GDDR6, 64MB L2)  
**Baseline TDP:** 250 W | **Power Capping Range:** [100 W, 250 W]  
**SLA Target:** Runtime Performance Degradation $\le 8.0\%$

---

## Summary of Results

| Workload | Category | Working Set | Regime | Exact AI (FLOP/B) | Golden Cap | Baseline Time | Gov Time | Deg (%) | Baseline Power | Gov Power | Net Energy Saved (%) | SLA Met? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `19_ReLU` | Memory-Bound (Activation) | 512.0 MB | GDDR DRAM | 0.25 | **120W** | 3.002s | 3.0s | -0.05% | 151.0W | 124.2W | **+17.77%** | **PASS** |
| `21_Sigmoid` | Memory-Bound (Activation) | 512.0 MB | GDDR DRAM | 0.25 | **120W** | 3.015s | 3.016s | +0.04% | 162.5W | 122.0W | **+24.91%** | **PASS** |
| `22_Tanh` | Memory-Bound (Activation) | 512.0 MB | GDDR DRAM | 0.25 | **120W** | 2.998s | 3.0s | +0.07% | 161.4W | 124.9W | **+22.59%** | **PASS** |
| `23_Softmax` | Memory-Bound (Activation) | 512.0 MB | GDDR DRAM | 0.25 | **120W** | 2.995s | 3.003s | +0.28% | 167.4W | 118.8W | **+28.83%** | **PASS** |
| `26_GELU_` | Memory-Bound (Activation) | 512.0 MB | GDDR DRAM | 0.25 | **120W** | 2.998s | 3.001s | +0.10% | 167.1W | 134.1W | **+19.69%** | **PASS** |
| `31_ELU` | Memory-Bound (Activation) | 512.0 MB | GDDR DRAM | 0.25 | **120W** | 3.002s | 3.006s | +0.13% | 159.7W | 122.4W | **+23.27%** | **PASS** |
| `10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh` | Compute-Bound (GEMM/Conv) | 2048.17 MB | GDDR DRAM | 143.99 | **250W** | 2.996s | 2.997s | +0.05% | 204.2W | 185.4W | **+9.14%** | **PASS** |
| `11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm` | Compute-Bound (GEMM/Conv) | 201.03 MB | GDDR DRAM | 510.54 | **250W** | 2.995s | 3.002s | +0.23% | 205.1W | 210.4W | **-2.84%** | **PASS** |
| `12_Gemm_Multiply_LeakyReLU` | Compute-Bound (GEMM/Conv) | 288.03 MB | GDDR DRAM | 113.77 | **250W** | 2.952s | 2.975s | +0.80% | 212.2W | 208.3W | **+1.05%** | **PASS** |
| `13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling` | Compute-Bound (GEMM/Conv) | 576.11 MB | GDDR DRAM | 383.93 | **250W** | 2.977s | 2.98s | +0.09% | 191.0W | 194.3W | **-1.82%** | **PASS** |
| `14_Gemm_Divide_Sum_Scaling` | Compute-Bound (GEMM/Conv) | 272.0 MB | GDDR DRAM | 120.47 | **250W** | 3.494s | 3.51s | +0.46% | 217.7W | 231.0W | **-6.63%** | **PASS** |
| `L2_Resident_GEMM_1024` | Compute-Bound (L2-Resident GEMM) | 6.0 MB | L2 Cache | 42.67 | **220W** | 2.562s | 2.66s | +3.84% | 197.9W | 186.1W | **+2.37%** | **PASS** |

---

## Key Findings

1. **Overall SLA Compliance:** **100.0%** of evaluated workloads strictly satisfied the $\le 8.0\%$ latency SLA.
2. **Memory-Bound Workloads:** Average energy savings of **+22.84%** (up to +28.83%) at Golden Caps of **120W**, with average degradation of only **+0.10%**.
3. **Compute-Bound Workloads:** Maintained **100% SLA compliance** by preserving 250W for DRAM-streaming GEMMs/Convolutions, while exploiting the 220W/230W compute plateau for L2-resident kernels.
4. **No Clock Locking:** GPU core clocks remained completely dynamic and unconstrained under NVIDIA GPU Boost within the allocated power cap.
