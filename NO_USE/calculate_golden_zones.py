import os
import csv
from collections import defaultdict

def analyze_golden_zone(file_path):
    # Read data
    data = defaultdict(list)
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or 'clock_target_mhz' not in row or not row['clock_target_mhz']:
                    continue
                try:
                    clock = int(row['clock_target_mhz'])
                    runtime = float(row['runtime_s'])
                    power = float(row['avg_power_w'])
                    data[clock].append((runtime, power))
                except ValueError:
                    continue
    except Exception as e:
        return None
                
    if not data:
        return None
        
    # Aggregate (average) by clock target
    aggregated = []
    for clock, runs in data.items():
        avg_runtime = sum(r[0] for r in runs) / len(runs)
        avg_power = sum(r[1] for r in runs) / len(runs)
        aggregated.append({'clock': clock, 'runtime': avg_runtime, 'power': avg_power})
        
    # Sort by clock descending to start from the baseline (max clock)
    aggregated.sort(key=lambda x: x['clock'], reverse=True)
    
    baseline = aggregated[0]
    baseline_runtime = baseline['runtime']
    # Define the 6% degradation threshold mathematically
    threshold_runtime = baseline_runtime * 1.06  
    
    golden_zone = baseline
    
    # Sweep downwards from max clock to find the lowest clock within the threshold
    for entry in aggregated:
        if entry['runtime'] <= threshold_runtime:
            golden_zone = entry
        else:
            # Once we cross the 6% performance drop threshold, stop looking
            break
            
    baseline_energy = baseline['power'] * baseline_runtime
    golden_energy = golden_zone['power'] * golden_zone['runtime']
    energy_saved_j = baseline_energy - golden_energy
    energy_saved_pct = (energy_saved_j / baseline_energy) * 100.0 if baseline_energy > 0 else 0.0

    return {
        'benchmark': os.path.basename(file_path),
        'baseline_clock': baseline['clock'],
        'baseline_runtime': baseline_runtime,
        'baseline_power': baseline['power'],
        'baseline_energy_j': baseline_energy,
        'golden_clock': golden_zone['clock'],
        'golden_runtime': golden_zone['runtime'],
        'golden_power': golden_zone['power'],
        'golden_energy_j': golden_energy,
        'power_saved_w': baseline['power'] - golden_zone['power'],
        'energy_saved_j': energy_saved_j,
        'energy_saved_pct': energy_saved_pct,
        'perf_drop_pct': ((golden_zone['runtime'] / baseline_runtime) - 1) * 100
    }

def main():
    base_dir = r"c:\Users\Darsh\Desktop\Goldenzone extension\Darsh Laptop Readings"
    categories = ['compute', 'memory']
    
    print("Calculating Golden Zones (Max 6% Performance Drop)...")
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        print(f"\n{'='*60}")
        print(f"ANALYZING CATEGORY: {category.upper()}")
        print(f"{'='*60}")
        
        if not os.path.exists(cat_dir):
            print(f"Directory not found: {cat_dir}")
            continue
            
        csv_files = [f for f in os.listdir(cat_dir) if f.endswith('.csv')]
        csv_files.sort()
        
        for file in csv_files:
            result = analyze_golden_zone(os.path.join(cat_dir, file))
            if result:
                print(f"Benchmark: {result['benchmark']}")
                print(f"  Baseline (No Cap): {result['baseline_clock']} MHz | {result['baseline_power']:.1f} W | {result['baseline_runtime']:.3f} s | Energy: {result['baseline_energy_j']:.1f} J")
                print(f"  Golden Zone      : {result['golden_clock']} MHz | {result['golden_power']:.1f} W | {result['golden_runtime']:.3f} s | Energy: {result['golden_energy_j']:.1f} J")
                print(f"  -> Energy Saved  : {result['energy_saved_j']:>6.1f} J ({result['energy_saved_pct']:>5.2f} %)")
                print(f"  -> Power Saved   : {result['power_saved_w']:>5.1f} W")
                print(f"  -> Perf Drop     : {result['perf_drop_pct']:>5.2f} %")
                print("-" * 60)

if __name__ == "__main__":
    main()
