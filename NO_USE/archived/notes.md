# Goldenzone Extension - Architectural Notes & Decisions

This document tracks the core logic and critical decisions made during the development of the Goldenzone Extension to ensure we don't lose context.

## 1. The Dual "Golden Zone" Strategy
*   **Hypothesis Validated:** Hardware telemetry proves that Memory and Compute tasks react entirely differently to power throttling.
*   **Compute-Bound:** Highly sensitive to clock speeds. Dropping the power cap causes linear performance drops. The Golden Zone (max 6% performance drop) is higher up the curve (e.g., 1785 - 2325 MHz), saving ~5-10W.
*   **Memory-Bound:** Highly resilient. The GPU math cores spend most of their time waiting for data from VRAM. The Golden Zone can be pushed aggressively low (e.g., 945 MHz), saving ~15-20W with virtually 0% performance loss.

## 2. The "Active Profiling Sweep" Model
Instead of a simple static snapshot, the ML Model evaluates a workload by looking at its **Reaction Curve**.
*   **In Production:** When a new app starts, the daemon rapidly sweeps through descending power caps (e.g., spending 2 seconds at each step). 
*   **Inference:** The model takes this array of telemetry and classifies the shape of the degradation. Compute tasks show high `gpu_util` and linear slowdowns. Memory tasks show high `mem_bw_util` and flat performance curves.

## 3. Training on Summary CSVs vs. Deep Logs
**Decision:** We train the model using the aggregated summary CSVs (`01_sgemm_nn.csv`), NOT the deep time-series logs located in the `logs/` folders.
*   **Reasoning:** Deep logs vary wildly in length. A benchmark at 3105 MHz might generate 27 rows of polling data, while the same benchmark throttled to 525 MHz generates 97 rows. ML models require fixed-length feature vectors. 
*   **The Fix:** The summary CSVs contain the *average* telemetry for each clock step. This is mathematically perfect because it exactly mirrors what our production daemon will do: poll the GPU for 2 seconds at a specific power cap, **average the results**, and feed that single vector step into the model.

## 4. Excluding `runtime_s` and `energy_j` from Training Features
**Decision:** The model is trained strictly on instantaneous telemetry (`avg_clock_mhz`, `avg_gpu_util_pct`, `avg_mem_bw_util_pct`, `avg_temp_c`, `avg_power_w`, etc.). We explicitly ban `runtime_s` and `energy_j` from the `FEATURE_COLS`.
*   **Reasoning:** During a real-world, real-time power sweep, the background daemon operates on a fixed timer (e.g., 2 seconds per power step). It does not wait for a "task to finish", so it cannot measure total `runtime_s` or total `energy_j`.
*   **The Fix:** If the model trained on `runtime_s`, it would perfectly separate the data by "cheating", but it would immediately fail during inference because the daemon cannot provide `runtime_s` as an input feature.

## 5. Preventing the VRAM "Data Leak"
**Decision:** We also explicitly banned `avg_vram_used_mib` and `avg_vram_alloc_pct` from the feature set.
*   **Reasoning:** During early evaluation on KBench workloads, the model classified memory-bound scripts (like `Sigmoid` or `LeakyReLU`) as Compute-bound whenever we artificially shrank the tensor size to fit on a 6GB GPU. 
*   **The "Aha!" Moment:** The Random Forest model had learned a false correlation. Because the training data used huge tensors for Memory benchmarks (~6GB VRAM) and small tensors for Compute benchmarks (~1.5GB VRAM), the model simply looked at the total memory allocation to classify the workload! It ignored the Power and Utilization curves completely.
*   **The Fix:** By removing VRAM allocation sizes from the training data, the model is forced to actually evaluate the physical behavior of the GPU (Power, Clock, Core Util, and Memory Bandwidth Util) to make its classification.
