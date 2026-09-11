import os
import sys
import time
import subprocess
import threading
import importlib.util
import torch
import joblib
import numpy as np
import csv
from io import StringIO

def set_gpu_clock(clock_mhz):
    try:
        # Requires Admin privileges on Windows
        subprocess.run(["nvidia-smi", "-lgc", f"{clock_mhz},{clock_mhz}"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def reset_gpu_clock():
    try:
        subprocess.run(["nvidia-smi", "-rgc"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass

def poll_nvidia_smi(duration_s=2):
    """ Polls nvidia-smi for 'duration_s' seconds and averages the 7 features """
    metrics = []
    start_time = time.time()
    
    query = "clocks.current.graphics,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    
    while time.time() - start_time < duration_s:
        try:
            res = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                text=True
            )
            f = StringIO(res.strip())
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 7:
                    clock = float(row[0])
                    temp = float(row[1])
                    gpu_util = float(row[2])
                    mem_util = float(row[3])
                    mem_used = float(row[4])
                    mem_total = float(row[5])
                    power = float(row[6])
                    
                    mem_alloc_pct = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                    
                    # We only append the 5 features the ML model expects, ignoring VRAM
                    metrics.append([clock, temp, gpu_util, mem_util, power])
        except Exception as e:
            time.sleep(0.1)
            continue
            
        time.sleep(0.5) 
        
    if not metrics:
        return [0.0]*5
        
    return np.mean(metrics, axis=0).tolist()

def run_workload_thread(module_path, stop_event):
    try:
        spec = importlib.util.spec_from_file_location("workload", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Instantiate model with init args if they exist
        if hasattr(module, 'get_init_inputs'):
            model = module.Model(*module.get_init_inputs()).cuda()
        else:
            model = module.Model().cuda()
            
        model.eval()
        
        # Prevent OOM on 6GB GPUs by forcing a smaller batch size
        if hasattr(module, 'batch_size') and module.batch_size > 512:
            module.batch_size = 512
            
        # Get inputs
        if hasattr(module, 'get_inputs'):
            x_list = [x.cuda() for x in module.get_inputs()]
        else:
            # Fallback if no get_inputs is defined
            batch_size = getattr(module, 'batch_size', 512)
            x_list = [torch.randn((batch_size, 1024), device='cuda')]
        
        # Infinite loop to keep GPU busy
        with torch.no_grad():
            while not stop_event.is_set():
                model(*x_list)
                torch.cuda.synchronize()
                
    except Exception as e:
        print(f"Error in workload thread for {module_path}: {e}")

def evaluate_workload(file_path, model, metadata):
    print(f"\nEvaluating: {os.path.basename(file_path)}")
    stop_event = threading.Event()
    
    # Start the workload in the background
    worker = threading.Thread(target=run_workload_thread, args=(file_path, stop_event))
    worker.start()
    
    # Give it a second to spin up and load data to VRAM
    time.sleep(2)
    
    feature_vector = []
    common_clocks = metadata['common_clocks']
    
    admin_warning_shown = False
    
    try:
        # Perform the Active Sweep
        for clock in common_clocks:
            success = set_gpu_clock(clock)
            if not success and not admin_warning_shown:
                print("⚠️ WARNING: Failed to lock GPU clock. You likely need to run this script as Administrator.")
                admin_warning_shown = True
            
            # Wait for clock to settle
            time.sleep(1)
            
            # Poll for 2 seconds
            metrics = poll_nvidia_smi(duration_s=2)
            feature_vector.extend(metrics)
            
            print(f"  Swept {clock} MHz | GPU: {metrics[2]:.1f}% | MemBW: {metrics[3]:.1f}% | Power: {metrics[4]:.1f}W")
            
        # Classify
        X = np.array(feature_vector).reshape(1, -1)
        pred = model.predict(X)[0]
        
        label = "Compute" if pred == 1 else "Memory"
        print(f"✅ Prediction for {os.path.basename(file_path)}: ** {label} **")
        
    finally:
        # Cleanup
        stop_event.set()
        worker.join(timeout=2)
        reset_gpu_clock()

def main():
    if not os.path.exists("workload_classifier.pkl"):
        print("Model not found. Run train_model.py first.")
        return
        
    model = joblib.load("workload_classifier.pkl")
    metadata = joblib.load("model_metadata.pkl")
    
    base_dir = r"c:\Users\Darsh\Desktop\Goldenzone extension\workloads_KBENCH_EVAL"
    categories = ['compute', 'memory']
    
    print(f"Loaded model. Expected {len(metadata['common_clocks'])} clock steps per sweep.")
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        files = [f for f in os.listdir(cat_dir) if f.endswith('.py')]
        for file in files:
            evaluate_workload(os.path.join(cat_dir, file), model, metadata)

if __name__ == "__main__":
    main()
