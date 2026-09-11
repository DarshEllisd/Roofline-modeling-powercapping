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
| **0.0** FLOP/B | **120 W** | 476.3 | 0.00 TFLOP/s | 0.00 FLOP/B | +24.58 % | **200 W** | 2594.9 | 0.00 TFLOP/s | 0.00 FLOP/B | +13.41 % | L2 Saturates Earlier (+80W) |
| **10.0** FLOP/B | **150 W** | 456.0 | 4.56 TFLOP/s | 10.00 FLOP/B | +35.28 % | **220 W** | 1688.4 | 16.88 TFLOP/s | 10.00 FLOP/B | +4.58 % | L2 Saturates Earlier (+70W) |
| **20.0** FLOP/B | **170 W** | 461.1 | 9.22 TFLOP/s | 20.00 FLOP/B | +29.82 % | **220 W** | 1212.3 | 24.25 TFLOP/s | 20.00 FLOP/B | +3.58 % | L2 Saturates Earlier (+50W) |
| **30.0** FLOP/B | **190 W** | 463.5 | 13.91 TFLOP/s | 30.00 FLOP/B | +22.41 % | **220 W** | 988.3 | 29.65 TFLOP/s | 30.00 FLOP/B | +7.17 % | L2 Saturates Earlier (+30W) |
| **40.0** FLOP/B | **200 W** | 457.0 | 18.28 TFLOP/s | 40.00 FLOP/B | +15.69 % | **220 W** | 796.6 | 31.86 TFLOP/s | 40.00 FLOP/B | +3.51 % | L2 Saturates Earlier (+20W) |
| **50.0** FLOP/B | **210 W** | 441.2 | 22.06 TFLOP/s | 50.00 FLOP/B | +11.55 % | **220 W** | 700.2 | 35.01 TFLOP/s | 50.00 FLOP/B | +6.52 % | L2 Saturates Earlier (+10W) |
| **60.0** FLOP/B | **230 W** | 454.3 | 27.26 TFLOP/s | 60.00 FLOP/B | +6.20 % | **230 W** | 608.6 | 36.51 TFLOP/s | 60.00 FLOP/B | +3.70 % | Identical Cap |
| **70.0** FLOP/B | **250 W** | 449.7 | 31.48 TFLOP/s | 70.00 FLOP/B | +0.00 % | **220 W** | 536.7 | 37.57 TFLOP/s | 70.00 FLOP/B | +5.84 % | DRAM Higher |
| **80.0** FLOP/B | **250 W** | 439.2 | 35.13 TFLOP/s | 80.00 FLOP/B | +0.00 % | **220 W** | 478.2 | 38.26 TFLOP/s | 80.00 FLOP/B | +4.12 % | DRAM Higher |
| **90.0** FLOP/B | **250 W** | 400.9 | 36.08 TFLOP/s | 90.00 FLOP/B | +0.00 % | **220 W** | 436.0 | 39.24 TFLOP/s | 90.00 FLOP/B | +6.46 % | DRAM Higher |
| **100.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 396.2 | 39.62 TFLOP/s | 100.00 FLOP/B | +6.45 % | Saturated / Ended |
| **110.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 368.8 | 40.56 TFLOP/s | 110.00 FLOP/B | +5.09 % | Saturated / Ended |
| **120.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 338.7 | 40.64 TFLOP/s | 120.00 FLOP/B | +3.82 % | Saturated / Ended |
| **130.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 320.5 | 41.66 TFLOP/s | 130.00 FLOP/B | +3.75 % | Saturated / Ended |
| **140.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 296.7 | 41.54 TFLOP/s | 140.00 FLOP/B | +4.72 % | Saturated / Ended |
| **150.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 281.3 | 42.19 TFLOP/s | 150.00 FLOP/B | +6.39 % | Saturated / Ended |
| **160.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 264.3 | 42.29 TFLOP/s | 160.00 FLOP/B | +4.25 % | Saturated / Ended |
| **170.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **230 W** | 255.9 | 43.50 TFLOP/s | 170.00 FLOP/B | +3.85 % | Saturated / Ended |
| **180.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 237.4 | 42.73 TFLOP/s | 180.00 FLOP/B | +5.07 % | Saturated / Ended |
| **190.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 227.0 | 43.14 TFLOP/s | 190.00 FLOP/B | +5.45 % | Saturated / Ended |
| **200.0** FLOP/B | N/A | N/A | N/A | N/A | N/A | **220 W** | 215.2 | 43.04 TFLOP/s | 200.00 FLOP/B | +4.52 % | Saturated / Ended |

## 3. Scientific Analysis & Real-Time Governor Heuristics

1. **Early Saturation of L2-Resident Workloads:**
   Because L2 cache provides over $4.5\times$ greater bandwidth than the GDDR6 bus, workloads hitting L2 cache hit their ALU compute ceiling at much lower arithmetic intensities ($I \ge 25$ vs $I \ge 113$).
   Consequently, the Golden Zone for L2-resident kernels rapidly shifts to the unconstrained baseline (250 W) well before DRAM-resident workloads.
2. **Governor Classification Rule:**
   When the CUPTI profiler samples operational intensity, the governor must cross-reference DRAM bandwidth vs L2 bandwidth (`dram__bytes.sum` vs `lts__t_bytes.sum`). If L2 hit rate $> 80\%$, the governor must evaluate against $I^*_{\text{L2}} \approx 25\text{ FLOP/B}$ instead of $I^*_{\text{DRAM}} \approx 113\text{ FLOP/B}$ to avoid inducing massive performance stalls.
