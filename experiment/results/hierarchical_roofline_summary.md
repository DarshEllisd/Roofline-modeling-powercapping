# Hierarchical Real-Time Roofline: GDDR DRAM vs. L2 Cache Golden Zones

**Target Hardware:** NVIDIA RTX 5000 Ada Generation (AD102, CC 8.9)
**Tolerance Threshold:** $\le 5.0\%$ Performance Degradation

## 1. Architectural Constants & Ridge Point Divergence

| Parameter | GDDR6 DRAM | L2 Cache (On-Chip Crossbar) |
| :--- | :--- | :--- |
| **Theoretical Peak Bandwidth ($B_{\text{peak}}$)** | **$576.0\text{ GB/s}$** | **$\sim 2,600.0\text{ GB/s}$** ($\approx 4.5\times$ higher) |
| **Compute Peak ($P_{\text{peak}}$)** | $65.28\text{ TFLOP/s}$ | $65.28\text{ TFLOP/s}$ |
| **Hardware Ridge Point ($I^*$)** | **$113.33\text{ FLOPs/Byte}$** | **$\approx 25.10\text{ FLOPs/Byte}$** |
| **Working Set Tested** | $256\text{ MB}$ (Flushes $64\text{ MB}$ L2 cache) | $16\text{ MB}$ (Resides $100\%$ inside L2 cache) |

## 2. Side-by-Side Golden Zone Comparison

| Arithmetic Intensity | GDDR DRAM Golden Cap | DRAM Energy Saved | L2 Cache Golden Cap | L2 Energy Saved | Architectural Divergence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0** FLOP/B | **100 W** | +24.02 % | **230 W** | +17.25 % | L2 Saturates Earlier (+130W) |
| **10.0** FLOP/B | **160 W** | +32.33 % | **250 W** | +0.00 % | L2 Saturates Earlier (+90W) |
| **20.0** FLOP/B | **170 W** | +29.51 % | **240 W** | +3.89 % | L2 Saturates Earlier (+70W) |
| **30.0** FLOP/B | **190 W** | +27.92 % | **230 W** | +3.93 % | L2 Saturates Earlier (+40W) |
| **40.0** FLOP/B | **200 W** | +13.86 % | **230 W** | +3.46 % | L2 Saturates Earlier (+30W) |
| **50.0** FLOP/B | **220 W** | +19.94 % | **240 W** | +5.13 % | L2 Saturates Earlier (+20W) |
| **60.0** FLOP/B | **250 W** | +0.00 % | **230 W** | +9.96 % | DRAM Higher |
| **70.0** FLOP/B | **250 W** | +0.00 % | **230 W** | +7.89 % | DRAM Higher |
| **80.0** FLOP/B | **250 W** | +0.00 % | **250 W** | +0.00 % | Identical Cap |
| **90.0** FLOP/B | N/A | N/A | **250 W** | +0.00 % | Saturated / Ended |
| **100.0** FLOP/B | N/A | N/A | **250 W** | +0.00 % | Saturated / Ended |

## 3. Scientific Analysis & Real-Time Governor Heuristics

1. **Early Saturation of L2-Resident Workloads:**
   Because L2 cache provides over $4.5\times$ greater bandwidth than the GDDR6 bus, workloads hitting L2 cache hit their ALU compute ceiling at much lower arithmetic intensities ($I \ge 25$ vs $I \ge 113$).
   Consequently, the Golden Zone for L2-resident kernels rapidly shifts to the unconstrained baseline (250 W) well before DRAM-resident workloads.
2. **Governor Classification Rule:**
   When the CUPTI profiler samples operational intensity, the governor must cross-reference DRAM bandwidth vs L2 bandwidth (`dram__bytes.sum` vs `lts__t_bytes.sum`). If L2 hit rate $> 80\%$, the governor must evaluate against $I^*_{\text{L2}} \approx 25\text{ FLOP/B}$ instead of $I^*_{\text{DRAM}} \approx 113\text{ FLOP/B}$ to avoid inducing massive performance stalls.
