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
| **0.0** FLOP/B | **100 W** | 476.5 | 0.00 TFLOP/s | 0.00 FLOP/B | +32.52 % | **190 W** | 2526.8 | 0.00 TFLOP/s | 0.00 FLOP/B | +17.60 % | L2 Saturates Earlier (+90W) |
| **10.0** FLOP/B | **150 W** | 451.4 | 4.51 TFLOP/s | 10.00 FLOP/B | +36.74 % | **210 W** | 1646.0 | 16.46 TFLOP/s | 10.00 FLOP/B | +9.93 % | L2 Saturates Earlier (+60W) |
| **20.0** FLOP/B | **170 W** | 460.4 | 9.21 TFLOP/s | 20.00 FLOP/B | +30.48 % | **210 W** | 1178.3 | 23.57 TFLOP/s | 20.00 FLOP/B | +9.76 % | L2 Saturates Earlier (+40W) |
| **30.0** FLOP/B | **190 W** | 463.3 | 13.90 TFLOP/s | 30.00 FLOP/B | +22.95 % | **210 W** | 962.1 | 28.86 TFLOP/s | 30.00 FLOP/B | +9.84 % | L2 Saturates Earlier (+20W) |
| **40.0** FLOP/B | **200 W** | 449.7 | 17.99 TFLOP/s | 40.00 FLOP/B | +16.75 % | **210 W** | 772.3 | 30.89 TFLOP/s | 40.00 FLOP/B | +9.43 % | L2 Saturates Earlier (+10W) |
| **50.0** FLOP/B | **220 W** | 457.1 | 22.85 TFLOP/s | 50.00 FLOP/B | +10.29 % | **210 W** | 680.6 | 34.03 TFLOP/s | 50.00 FLOP/B | +9.56 % | DRAM Higher |
| **60.0** FLOP/B | **240 W** | 456.9 | 27.41 TFLOP/s | 60.00 FLOP/B | +4.87 % | **210 W** | 580.0 | 34.80 TFLOP/s | 60.00 FLOP/B | +9.54 % | DRAM Higher |
| **70.0** FLOP/B | **250 W** | 445.4 | 31.18 TFLOP/s | 70.00 FLOP/B | +0.00 % | **210 W** | 521.4 | 36.50 TFLOP/s | 70.00 FLOP/B | +9.59 % | DRAM Higher |
| **80.0** FLOP/B | **250 W** | 411.5 | 32.92 TFLOP/s | 80.00 FLOP/B | +0.00 % | **210 W** | 465.7 | 37.25 TFLOP/s | 80.00 FLOP/B | +9.57 % | DRAM Higher |
| **90.0** FLOP/B | **250 W** | 371.9 | 33.47 TFLOP/s | 90.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **100.0** FLOP/B | **250 W** | 346.0 | 34.60 TFLOP/s | 100.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **110.0** FLOP/B | **250 W** | 321.8 | 35.40 TFLOP/s | 110.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **120.0** FLOP/B | **250 W** | 301.6 | 36.19 TFLOP/s | 120.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |
| **130.0** FLOP/B | **250 W** | 285.8 | 37.15 TFLOP/s | 130.00 FLOP/B | +0.00 % | N/A | N/A | N/A | N/A | N/A | Saturated / Ended |

## 3. Scientific Analysis & Real-Time Governor Heuristics

1. **Early Saturation of L2-Resident Workloads:**
   Because L2 cache provides over $4.5\times$ greater bandwidth than the GDDR6 bus, workloads hitting L2 cache hit their ALU compute ceiling at much lower arithmetic intensities ($I \ge 25$ vs $I \ge 113$).
   Consequently, the Golden Zone for L2-resident kernels rapidly shifts to the unconstrained baseline (250 W) well before DRAM-resident workloads.
2. **Governor Classification Rule:**
   When the CUPTI profiler samples operational intensity, the governor must cross-reference DRAM bandwidth vs L2 bandwidth (`dram__bytes.sum` vs `lts__t_bytes.sum`). If L2 hit rate $> 80\%$, the governor must evaluate against $I^*_{\text{L2}} \approx 25\text{ FLOP/B}$ instead of $I^*_{\text{DRAM}} \approx 113\text{ FLOP/B}$ to avoid inducing massive performance stalls.
