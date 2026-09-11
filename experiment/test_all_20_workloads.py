import os
import sys
import glob
import torch
import importlib.util

sys.path.insert(0, os.path.abspath("experiment"))
from evaluate_kbench_workloads import compute_exact_workload_metrics, get_golden_zone_power_cap_8pct, load_workload

torch.cuda.init()

base_dir = os.path.abspath(".")
kbench_dir = os.path.join(base_dir, "workloads_KBENCH_EVAL")
mem_dir = os.path.join(kbench_dir, "memory")
comp_dir = os.path.join(kbench_dir, "compute")

workloads = []
for f in sorted(glob.glob(os.path.join(mem_dir, "*.py"))):
    workloads.append((f, "Memory-Bound (Activation)"))
for f in sorted(glob.glob(os.path.join(comp_dir, "*.py"))):
    workloads.append((f, "Compute-Bound (GEMM/Conv)"))
workloads.append(("L2_Cache_Resident_GEMM", "Compute-Bound (L2-Resident GEMM)"))

print(f"Total workloads discovered: {len(workloads)}")

for idx, (fpath, cat) in enumerate(workloads, 1):
    wname = "L2_Resident_GEMM_1024" if fpath == "L2_Cache_Resident_GEMM" else os.path.basename(fpath).replace(".py", "")
    try:
        if fpath == "L2_Cache_Resident_GEMM":
            import torch.nn as nn
            class L2GemmModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(1024, 1024, bias=False)
                def forward(self, x):
                    return self.linear(x)
            model = L2GemmModel().cuda().eval()
            inputs = [torch.randn(256, 1024, device='cuda')]
        else:
            mod = load_workload(fpath)
            if hasattr(mod, 'batch_size') and mod.batch_size > 512:
                mod.batch_size = 512
            if hasattr(mod, 'dim') and mod.dim > 131072:
                mod.dim = 131072
            init_inputs = mod.get_init_inputs() if hasattr(mod, 'get_init_inputs') else []
            model = mod.Model(*init_inputs).cuda().eval()
            raw_inputs = mod.get_inputs() if hasattr(mod, 'get_inputs') else [torch.randn(512, 1024)]
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in raw_inputs]

        # Test single forward
        with torch.no_grad():
            out = model(*inputs)
        torch.cuda.synchronize()

        ws_mb, flops, ai, is_l2 = compute_exact_workload_metrics(model, inputs)
        golden_cap = get_golden_zone_power_cap_8pct(ai, is_l2)
        regime = "L2" if is_l2 else "DRAM"
        print(f"[{idx:02d}/20] OK: {wname[:35]:35s} | {regime:4s} | WS: {ws_mb:6.1f}MB | FLOPs: {flops:.2e} | AI: {ai:6.2f} | Cap: {golden_cap}W")
    except Exception as e:
        print(f"[{idx:02d}/20] ERROR: {wname}: {e}")
