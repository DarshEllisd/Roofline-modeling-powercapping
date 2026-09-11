import os
import csv
import re
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# The features we want to extract at each clock step during the sweep
FEATURE_COLS = [
    'avg_clock_mhz', 'avg_temp_c', 'avg_gpu_util_pct', 
    'avg_mem_bw_util_pct', 'avg_power_w'
]

def extract_sweeps_from_csv(file_path):
    """
    Parses a benchmark CSV file and separates the runs (r1, r2, r3) into distinct sweeps.
    Returns: dict of runs, where runs['r1'] = {3105: [feats], 3045: [feats], ...}
    """
    runs = defaultdict(dict)
    
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or 'clock_target_mhz' not in row or not row['clock_target_mhz']:
                    continue
                try:
                    clock = int(row['clock_target_mhz'])
                    run_id = row['run_id']
                    
                    # Extract 'r1', 'r2', 'r3' from the run_id string
                    match = re.search(r'_(r\d+)_', run_id)
                    if match:
                        run_idx = match.group(1)
                    else:
                        continue
                        
                    feats = [float(row[col]) for col in FEATURE_COLS]
                    runs[run_idx][clock] = feats
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

    return runs

def get_common_clocks(all_data):
    """
    Finds the intersection of all clock targets across all runs and files 
    to ensure every feature vector is the exact same length.
    """
    clock_sets = []
    for category, files in all_data.items():
        for file_path, sweeps in files.items():
            for run_idx, clocks_dict in sweeps.items():
                clock_sets.append(set(clocks_dict.keys()))
                
    if not clock_sets:
        return []
        
    common_clocks = set.intersection(*clock_sets)
    return sorted(list(common_clocks), reverse=True)

def build_dataset(base_dir):
    """
    Reads the compute/memory folders and flattens the sequences into training arrays (X, y).
    """
    categories = {'memory': 0, 'compute': 1}
    all_data = defaultdict(lambda: defaultdict(dict))
    
    for category, label in categories.items():
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            print(f"Directory not found: {cat_dir}")
            continue
            
        for file in os.listdir(cat_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(cat_dir, file)
                sweeps = extract_sweeps_from_csv(file_path)
                if sweeps:
                    all_data[category][file] = sweeps
                    
    common_clocks = get_common_clocks(all_data)
    print(f"Found {len(common_clocks)} common clock steps across all datasets.")
    
    if not common_clocks:
        print("No common clock steps found. Cannot build dataset.")
        return [], [], []

    X = []
    y = []
    
    for category, label in categories.items():
        for file_path, sweeps in all_data[category].items():
            for run_idx, clocks_dict in sweeps.items():
                # Build the flattened feature vector for this sweep
                feature_vector = []
                missing_clock = False
                for clock in common_clocks:
                    if clock in clocks_dict:
                        feature_vector.extend(clocks_dict[clock])
                    else:
                        missing_clock = True
                        break
                        
                if not missing_clock:
                    X.append(feature_vector)
                    y.append(label)
                    
    return np.array(X), np.array(y), common_clocks

def main():
    base_dir = r"c:\Users\Darsh\Desktop\Goldenzone extension\Darsh Laptop Readings"
    
    print("Building dataset from CSV sweeps...")
    X, y, common_clocks = build_dataset(base_dir)
    
    if len(X) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    print(f"Dataset built successfully. Shape: {X.shape}")
    print(f"Total samples (sweeps): {len(y)} ({sum(y==1)} Compute, {sum(y==0)} Memory)")
    
    # We do a basic 80/20 train/test split here just to validate the model learns the training distribution well.
    # The true test will be the KBench EVAL files later.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining Random Forest Model...")
    # Very small model so inference takes <1ms in the background daemon
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nInternal Validation Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Memory (0)', 'Compute (1)']))
    
    # Save the model
    model_path = "workload_classifier.pkl"
    joblib.dump(clf, model_path)
    
    # Save metadata so the runtime daemon knows exactly what format the model expects
    metadata = {
        'common_clocks': common_clocks,
        'features_per_clock': FEATURE_COLS
    }
    joblib.dump(metadata, "model_metadata.pkl")
    
    print(f"\nModel successfully saved to {model_path}")
    print(f"Metadata saved to model_metadata.pkl")

if __name__ == "__main__":
    main()
