#!/usr/bin/env python3
"""
Report Generator for Hierarchical Roofline Golden Zones
Reads existing CSV results and produces hierarchical_roofline_summary.md
"""
import os
import pandas as pd

def generate():
    results_dir = "/home/antpc/Desktop/goldenzone/Roofline-modeling-powercapping/experiment/results"
    dram_csv = os.path.join(results_dir, "dram", "golden_zone_by_ai.csv")
    l2_csv = os.path.join(results_dir, "l2", "golden_zone_by_ai.csv")
    report_path = os.path.join(results_dir, "hierarchical_roofline_summary.md")

    dram_gz = pd.read_csv(dram_csv) if os.path.exists(dram_csv) else None
    l2_gz = pd.read_csv(l2_csv) if os.path.exists(l2_csv) else None

    all_ais = sorted(list(set(
        (dram_gz['target_ai'].tolist() if dram_gz is not None else []) +
        (l2_gz['target_ai'].tolist() if l2_gz is not None else [])
    )))

    dram_map = {r['target_ai']: r for _, r in dram_gz.iterrows()} if dram_gz is not None else {}
    l2_map = {r['target_ai']: r for _, r in l2_gz.iterrows()} if l2_gz is not None else {}

    with open(report_path, "w") as f:
        f.write("# Hierarchical Real-Time Roofline: GDDR DRAM vs. L2 Cache Golden Zones\n\n")
        f.write("**Target Hardware:** NVIDIA RTX 5000 Ada Generation (AD102, CC 8.9)\n")
        f.write("**Performance Degradation Tolerance:** $\\le 5.0\\%$\n\n")

        f.write("## 1. Architectural Constants & Ridge Point Divergence\n\n")
        f.write("| Parameter | GDDR6 DRAM | L2 Cache (On-Chip Crossbar) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| **Theoretical Peak Bandwidth ($B_{\\text{peak}}$)** | **$576.0\\text{ GB/s}$** | **$\\sim 2,600.0\\text{ GB/s}$** ($\\approx 4.5\\times$ higher) |\n")
        f.write("| **Compute Peak ($P_{\\text{peak}}$)** | $65.28\\text{ TFLOP/s}$ | $65.28\\text{ TFLOP/s}$ |\n")
        f.write("| **Hardware Ridge Point ($I^*$)** | **$113.33\\text{ FLOPs/Byte}$** | **$\\approx 25.10\\text{ FLOPs/Byte}$** |\n")
        f.write("| **Working Set Tested** | $256\\text{ MB}$ (Flushes $64\\text{ MB}$ L2 cache) | $16\\text{ MB}$ (Resides $100\\%$ inside L2 cache) |\n\n")

        f.write("## 2. Side-by-Side Golden Zone Comparison\n\n")
        f.write("| Arithmetic Intensity | GDDR DRAM Golden Cap | DRAM Energy Saved | L2 Cache Golden Cap | L2 Energy Saved | Architectural Divergence |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for ai in all_ais:
            d_row = dram_map.get(ai, None)
            l_row = l2_map.get(ai, None)

            d_cap_str = f"**{int(d_row['golden_cap_w'])} W**" if d_row is not None else "N/A"
            d_sav_str = f"{d_row['golden_energy_saved_pct']:>+5.2f} %" if d_row is not None else "N/A"
            
            l_cap_str = f"**{int(l_row['golden_cap_w'])} W**" if l_row is not None else "N/A"
            l_sav_str = f"{l_row['golden_energy_saved_pct']:>+5.2f} %" if l_row is not None else "N/A"

            if d_row is not None and l_row is not None:
                if d_row['golden_cap_w'] == l_row['golden_cap_w']:
                    div_str = "Identical Cap"
                elif d_row['golden_cap_w'] < l_row['golden_cap_w']:
                    div_str = f"L2 Saturates Earlier (+{int(l_row['golden_cap_w'] - d_row['golden_cap_w'])}W)"
                else:
                    div_str = "DRAM Higher"
            else:
                div_str = "Saturated / Ended"

            f.write(f"| **{ai:.1f}** FLOP/B | {d_cap_str} | {d_sav_str} | {l_cap_str} | {l_sav_str} | {div_str} |\n")

        f.write("\n## 3. Scientific Analysis & Real-Time Governor Heuristics\n\n")
        f.write("1. **Early Saturation of L2-Resident Workloads:**\n")
        f.write("   Because L2 cache provides over $4.5\\times$ greater bandwidth than the GDDR6 bus, workloads hitting L2 cache hit their ALU compute ceiling at much lower arithmetic intensities ($I \\ge 25$ vs $I \\ge 113$).\n")
        f.write("2. **Governor Classification Rule:**\n")
        f.write("   When the CUPTI profiler samples operational intensity, the governor must cross-reference DRAM bandwidth vs L2 bandwidth (`dram__bytes.sum` vs `lts__t_bytes.sum`). If L2 hit rate $> 80\\%$, the governor must evaluate against $I^*_{\\text{L2}} \\approx 25\\text{ FLOP/B}$ instead of $I^*_{\\text{DRAM}} \\approx 113\\text{ FLOP/B}$ to avoid inducing massive performance stalls.\n")

    print(f"Successfully generated: {report_path}")

if __name__ == "__main__":
    generate()
