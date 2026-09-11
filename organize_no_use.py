#!/usr/bin/env python3
"""
Move legacy, outdated, and unused files to NO_USE/ folder
Cleans up the repository so only active workstation components remain.
"""

import os
import shutil

def organize_workspace():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    no_use_dir = os.path.join(base_dir, "NO_USE")
    os.makedirs(no_use_dir, exist_ok=True)

    # Top-level obsolete files and folders from Windows laptop experiments
    obsolete_items = [
        "Darsh Laptop Readings",
        "archived",
        "readings.txt",
        "benchmark_switch_overhead.py",
        "calculate_golden_zones.py",
        "compile_intensity.bat",
        "intensity_kernels.cu",
        "intensity_kernels.dll",
        "test_cupti.cpp",
        "test_degradation_energy.py",
        "test_equidistant_degradation.py",
        "test_golden_governor.py",
        "test_ridge_point_degradation.py",
    ]

    for item in obsolete_items:
        src = os.path.join(base_dir, item)
        dst = os.path.join(no_use_dir, item)
        if os.path.exists(src):
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
            print(f"  [CLEANUP] Moved to NO_USE: {item}")

    # Obsolete files in roofline_plugin
    plugin_no_use = os.path.join(no_use_dir, "roofline_plugin_legacy")
    os.makedirs(plugin_no_use, exist_ok=True)
    plugin_dir = os.path.join(base_dir, "roofline_plugin")
    
    plugin_obsolete = [
        "compile.bat",
        "roofline_plugin.dll",
        "metrics.txt",
        "test_dll.py",
    ]

    for item in plugin_obsolete:
        src = os.path.join(plugin_dir, item)
        dst = os.path.join(plugin_no_use, item)
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            print(f"  [CLEANUP] Moved to NO_USE/roofline_plugin_legacy: {item}")

    print("\n[CLEANUP COMPLETE] All outdated files have been safely separated into NO_USE/.")

if __name__ == "__main__":
    organize_workspace()
