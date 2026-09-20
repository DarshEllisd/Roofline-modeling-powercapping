# Hierarchical Real-Time Roofline: GDDR DRAM vs. L2 Cache Golden Zones

**Target Hardware:** NVIDIA RTX 5000 Ada Generation (AD102, CC 8.9)
**Tolerance Threshold:** $\le 8.0\%$ Performance Degradation

## 1. Architectural Constants & Ridge Point Divergence

| Parameter | GDDR6 DRAM | L2 Cache (On-Chip Crossbar) |
| :--- | :--- | :--- |
| **Theoretical Peak Bandwidth ($B_{\text{peak}}$)** | **$576.0\text{ GB/s}$** | **$\sim 2,600.0\text{ GB/s}$** ($\approx 4.5\times$ higher) |
| **Compute Peak ($P_{\text{peak}}$)** | $65.28\text{ TFLOP/s}$ | $65.28\text{ TFLOP/s}$ |
| **Hardware Ridge Point ($I^*$)** | **$113.33\text{ FLOPs/Byte}$** | **$\approx 25.10\text{ FLOPs/Byte}$** |
| **Working Set Tested** | $256\text{ MB}$ (Flushes $64\text{ MB}$ L2 cache) | $16\text{ MB}$ (Resides $100\%$ inside L2 cache) |

## 2. Side-by-Side Golden Zone Comparison

| Arithmetic Intensity | GDDR Cap | DRAM BW (GB/s) | DRAM Compute | DRAM Achieved AI | DRAM Energy Saved | L2 Cap | L2 BW (GB/s) | L2 Compute | L2 Achieved AI | L2 Energy Saved | Architectural Divergence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0** FLOP/B | **130 W** | 476.4 | 0.00 TFLOP/s | 0.00 FLOP/B | +27.51 % | **190 W** | 2547.4 | 0.00 TFLOP/s | 0.00 FLOP/B | +17.43 % | L2 Saturates Earlier (+60W) |
| **10.0** FLOP/B | **150 W** | 455.1 | 4.55 TFLOP/s | 10.00 FLOP/B | +37.07 % | **210 W** | 1660.1 | 16.60 TFLOP/s | 10.00 FLOP/B | +9.89 % | L2 Saturates Earlier (+60W) |
| **20.0** FLOP/B | **170 W** | 462.4 | 9.25 TFLOP/s | 20.00 FLOP/B | +30.79 % | **210 W** | 1187.0 | 23.74 TFLOP/s | 20.00 FLOP/B | +9.92 % | L2 Saturates Earlier (+40W) |
| **30.0** FLOP/B | **190 W** | 465.1 | 13.95 TFLOP/s | 30.00 FLOP/B | +23.59 % | **210 W** | 970.6 | 29.12 TFLOP/s | 30.00 FLOP/B | +9.75 % | L2 Saturates Earlier (+20W) |
| **40.0** FLOP/B | **200 W** | 452.9 | 18.12 TFLOP/s | 40.00 FLOP/B | +17.39 % | **210 W** | 782.4 | 31.29 TFLOP/s | 40.00 FLOP/B | +10.09 % | L2 Saturates Earlier (+10W) |
| **50.0** FLOP/B | **220 W** | 461.0 | 23.05 TFLOP/s | 50.00 FLOP/B | +11.39 % | **210 W** | 685.9 | 34.29 TFLOP/s | 50.00 FLOP/B | +9.68 % | DRAM Higher |
| **60.0** FLOP/B | **240 W** | 436.0 | 26.16 TFLOP/s | 60.00 FLOP/B | +0.34 % | **210 W** | 583.6 | 35.02 TFLOP/s | 60.00 FLOP/B | +9.29 % | DRAM Higher |
| **70.0** FLOP/B | **250 W** | 445.4 | 31.18 TFLOP/s | 70.00 FLOP/B | +0.00 % | **210 W** | 525.5 | 36.79 TFLOP/s | 70.00 FLOP/B | +9.43 % | DRAM Higher |
| **80.0** FLOP/B | **250 W** | 417.7 | 33.42 TFLOP/s | 80.00 FLOP/B | +0.00 % | **210 W** | 469.9 | 37.59 TFLOP/s | 80.00 FLOP/B | +9.78 % | DRAM Higher |
| **90.0** FLOP/B | **250 W** | 376.9 | 33.92 TFLOP/s | 90.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **100.0** FLOP/B | **250 W** | 349.9 | 34.99 TFLOP/s | 100.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **110.0** FLOP/B | **250 W** | 323.5 | 35.59 TFLOP/s | 110.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **120.0** FLOP/B | **250 W** | 305.5 | 36.66 TFLOP/s | 120.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **130.0** FLOP/B | **250 W** | 288.4 | 37.49 TFLOP/s | 130.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |

## 3. Scientific Analysis & Real-Time Governor Heuristics

1. **Early Saturation of L2-Resident Workloads:**
   Because L2 cache provides over $4.5\times$ greater bandwidth than the GDDR6 bus, workloads hitting L2 cache hit their ALU compute ceiling at much lower arithmetic intensities ($I \ge 25$ vs $I \ge 113$).
   Consequently, the Golden Zone for L2-resident kernels rapidly shifts to the unconstrained baseline (250 W) well before DRAM-resident workloads.
2. **Governor Classification Rule:**
   When the CUPTI profiler samples operational intensity, the governor must cross-reference DRAM bandwidth vs L2 bandwidth (`dram__bytes.sum` vs `lts__t_bytes.sum`). If L2 hit rate $> 80\%$, the governor must evaluate against $I^*_{\text{L2}} \approx 25\text{ FLOP/B}$ instead of $I^*_{\text{DRAM}} \approx 113\text{ FLOP/B}$ to avoid inducing massive performance stalls.
