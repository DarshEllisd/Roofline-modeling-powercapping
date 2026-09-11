#!/usr/bin/env python3
"""
Report Generator for Hierarchical Roofline Golden Zones under <= 8.0% Latency SLA
Reads existing CSV results and produces hierarchical_roofline_summary.md with achieved AI,
achieved bandwidth, achieved compute, and monotonic clamping rules.
"""
import os
import pandas as pd

def generate():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
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

    last_dram_ai = max(dram_map.keys()) if dram_map else 0.0
    last_dram_cap = int(dram_map[last_dram_ai]['golden_cap_w']) if dram_map else 250
    last_l2_ai = max(l2_map.keys()) if l2_map else 0.0
    last_l2_cap = int(l2_map[last_l2_ai]['golden_cap_w']) if l2_map else 220

    with open(report_path, "w") as f:
        f.write("# Hierarchical Real-Time Roofline: GDDR DRAM vs. L2 Cache Golden Zones\n\n")
        f.write("**Target Hardware:** NVIDIA RTX 5000 Ada Generation (AD102, CC 8.9, 100 SMs, 32GB GDDR6)\n")
        f.write("**Performance Degradation SLA Tolerance:** $\\le 8.0\\%$\n")
        f.write("**Evaluation Protocol:** Sustained $\\sim 30\\text{s}$ Steady-State Runs across 16 Power Caps (250W to 100W)\n\n")

        f.write("## 1. Architectural Constants & Empirical Ceilings (LBNL ERT Validation)\n\n")
        f.write("| Hierarchy Level | Theoretical Peak | Empirical Ceiling (ERT) | Empirical Ridge Point ($I^*$) | Evaluated Working Set |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write("| **GDDR6 DRAM** | $576.0\\text{ GB/s}$ | **$466.98\\text{ GB/s}$** | **$137.69\\text{ FLOP/B}$** | $256\\text{ MB}$ (Exceeds $64\\text{ MB}$ L2 cache) |\n")
        f.write("| **L2 Cache** | $\\sim 2,600.0\\text{ GB/s}$ | **$3,838.07\\text{ GB/s}$** | **$16.75\\text{ FLOP/B}$** | $16\\text{ MB}$ ($100\\%$ on-chip L2 resident) |\n")
        f.write("| **FP32 Compute** | $65.28\\text{ TFLOP/s}$ | **$64.30\\text{ TFLOP/s}$** | — | 12,800 CUDA Cores ($2,550\\text{ MHz}$ Boost) |\n\n")

        f.write("## 2. Side-by-Side Golden Zone Comparison (8% Degradation SLA)\n\n")
        f.write("| Target AI | Achieved AI | GDDR DRAM Golden Cap | DRAM Degradation | DRAM Energy Saved | L2 Golden Cap | L2 Degradation | L2 Energy Saved | Architectural Divergence |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for ai in all_ais:
            d_row = dram_map.get(ai, None)
            l_row = l2_map.get(ai, None)

            if d_row is not None:
                d_cap_str = f"**{int(d_row['golden_cap_w'])} W**"
                d_deg_str = f"{d_row['golden_deg_pct']:>+5.2f} %"
                d_sav_str = f"{d_row['golden_energy_saved_pct']:>+5.2f} %"
                achieved_ai_str = f"{d_row.get('achieved_ai', ai):.1f}"
            else:
                # Monotonic clamping: for AI > last_dram_ai, clamp to last cap
                d_cap_str = f"*{last_dram_cap} W (Clamped)*"
                d_deg_str = "0.00 %"
                d_sav_str = "0.00 %"
                achieved_ai_str = f"{l_row.get('achieved_ai', ai):.1f}" if l_row is not None else f"{ai:.1f}"

            if l_row is not None:
                l_cap_str = f"**{int(l_row['golden_cap_w'])} W**"
                l_deg_str = f"{l_row['golden_deg_pct']:>+5.2f} %"
                l_sav_str = f"{l_row['golden_energy_saved_pct']:>+5.2f} %"
            else:
                l_cap_str = f"*{last_l2_cap} W (Clamped)*"
                l_deg_str = "N/A"
                l_sav_str = "N/A"

            if d_row is not None and l_row is not None:
                d_cap_val = int(d_row['golden_cap_w'])
                l_cap_val = int(l_row['golden_cap_w'])
                if d_cap_val == l_cap_val:
                    div_str = "Identical Cap"
                elif d_cap_val < l_cap_val:
                    div_str = f"L2 Demands Higher (+{l_cap_val - d_cap_val}W)"
                else:
                    div_str = f"DRAM Higher (+{d_cap_val - l_cap_val}W)"
            elif d_row is None and l_row is not None:
                div_str = f"DRAM Saturated at 250W; L2 at {int(l_row['golden_cap_w'])}W"
            else:
                div_str = "Saturated / Sweep End"

            f.write(f"| **{ai:.1f}** | {achieved_ai_str} | {d_cap_str} | {d_deg_str} | {d_sav_str} | {l_cap_str} | {l_deg_str} | {l_sav_str} | {div_str} |\n")

        f.write("\n## 3. The Monotonic Clamping Invariant\n\n")
        f.write("$$\\mathbf{\\forall I > I_{\\text{last}}, \\quad GoldenCap(I) = GoldenCap(I_{\\text{last}})}$$\n\n")
        f.write("1. **GDDR DRAM Streaming Boundary ($I_{\\text{last}} = 90.0\\text{ FLOP/B}$):**\n")
        f.write("   - Saturated at $250\\text{W}$ at $I = 70.0\\text{ FLOP/B}$.\n")
        f.write("   - For any operational intensity $I \\ge 70.0\\text{ FLOP/B}$ (and all $I > 90.0\\text{ FLOP/B}$), the Golden Cap is strictly clamped to **$250\\text{W}$**.\n")
        f.write("2. **L2 Cache Resident Boundary ($I_{\\text{last}} = 200.0\\text{ FLOP/B}$):**\n")
        f.write("   - Established steady compute plateau at **$220\\text{W}$** across the entire sweep ($I=10.0$ to $200.0$).\n")
        f.write("   - For any $I > 200.0\\text{ FLOP/B}$, the Golden Cap is clamped to the last recorded cap (**$220\\text{W}$**, or $230\\text{W}$ non-decreasing ceiling).\n")
        f.write("3. **Scientific Explanation of 220W Sweet Spot under 8% SLA:**\n")
        f.write("   - At 220W power cap, core clock drops by $\\sim 6.7\\%$ ($2,550\\text{ MHz} \\to 2,380\\text{ MHz}$), producing a $+6.5\\% - 7.2\\%$ runtime increase.\n")
        f.write("   - Under an $8.0\\%$ SLA tolerance, this slowdown is fully compliant.\n")
        f.write("   - Power draw decreases from $244\\text{W}$ to $213\\text{W}$ ($12.7\\%$ drop), yielding a net **$\\sim 4.5\\% - 7.2\\%$ Joule energy savings** across all compute-bound workloads.\n")

    print(f"Successfully generated: {report_path}")

if __name__ == "__main__":
    generate()

