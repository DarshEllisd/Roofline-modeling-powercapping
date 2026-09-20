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
| **0.0** FLOP/B | **100 W** | 476.5 | 0.00 TFLOP/s | 0.00 FLOP/B | +31.49 % | **190 W** | 2530.5 | 0.00 TFLOP/s | 0.00 FLOP/B | +17.58 % | L2 Saturates Earlier (+90W) |
| **10.0** FLOP/B | **150 W** | 453.8 | 4.54 TFLOP/s | 10.00 FLOP/B | +37.22 % | **210 W** | 1652.5 | 16.52 TFLOP/s | 10.00 FLOP/B | +10.21 % | L2 Saturates Earlier (+60W) |
| **20.0** FLOP/B | **170 W** | 460.0 | 9.20 TFLOP/s | 20.00 FLOP/B | +30.47 % | **210 W** | 1179.0 | 23.58 TFLOP/s | 20.00 FLOP/B | +9.68 % | L2 Saturates Earlier (+40W) |
| **30.0** FLOP/B | **190 W** | 462.1 | 13.86 TFLOP/s | 30.00 FLOP/B | +22.94 % | **210 W** | 961.8 | 28.85 TFLOP/s | 30.00 FLOP/B | +9.73 % | L2 Saturates Earlier (+20W) |
| **40.0** FLOP/B | **200 W** | 451.1 | 18.04 TFLOP/s | 40.00 FLOP/B | +17.24 % | **210 W** | 774.2 | 30.97 TFLOP/s | 40.00 FLOP/B | +9.50 % | L2 Saturates Earlier (+10W) |
| **50.0** FLOP/B | **220 W** | 459.3 | 22.97 TFLOP/s | 50.00 FLOP/B | +11.05 % | **210 W** | 681.4 | 34.07 TFLOP/s | 50.00 FLOP/B | +9.57 % | DRAM Higher |
| **60.0** FLOP/B | **240 W** | 457.4 | 27.44 TFLOP/s | 60.00 FLOP/B | +4.97 % | **210 W** | 580.1 | 34.81 TFLOP/s | 60.00 FLOP/B | +9.59 % | DRAM Higher |
| **70.0** FLOP/B | **250 W** | 443.6 | 31.05 TFLOP/s | 70.00 FLOP/B | +0.00 % | **210 W** | 521.3 | 36.49 TFLOP/s | 70.00 FLOP/B | +9.42 % | DRAM Higher |
| **80.0** FLOP/B | **250 W** | 413.0 | 33.04 TFLOP/s | 80.00 FLOP/B | +0.00 % | **210 W** | 466.1 | 37.29 TFLOP/s | 80.00 FLOP/B | +9.63 % | DRAM Higher |
| **90.0** FLOP/B | **250 W** | 372.5 | 33.52 TFLOP/s | 90.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **100.0** FLOP/B | **250 W** | 347.3 | 34.73 TFLOP/s | 100.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **110.0** FLOP/B | **250 W** | 322.2 | 35.44 TFLOP/s | 110.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **120.0** FLOP/B | **250 W** | 305.7 | 36.68 TFLOP/s | 120.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **130.0** FLOP/B | **250 W** | 289.4 | 37.63 TFLOP/s | 130.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |

## 3. Scientific Analysis & Real-Time Governor Heuristics

1. **Early Saturation of L2-Resident Workloads:**
   Because L2 cache provides over $4.5\times$ greater bandwidth than the GDDR6 bus, workloads hitting L2 cache hit their ALU compute ceiling at much lower arithmetic intensities ($I \ge 25$ vs $I \ge 113$).
   Consequently, the Golden Zone for L2-resident kernels rapidly shifts to the unconstrained baseline (250 W) well before DRAM-resident workloads.
2. **Governor Classification Rule:**
   When the CUPTI profiler samples operational intensity, the governor must cross-reference DRAM bandwidth vs L2 bandwidth (`dram__bytes.sum` vs `lts__t_bytes.sum`). If L2 hit rate $> 80\%$, the governor must evaluate against $I^*_{\text{L2}} \approx 25\text{ FLOP/B}$ instead of $I^*_{\text{DRAM}} \approx 113\text{ FLOP/B}$ to avoid inducing massive performance stalls.
