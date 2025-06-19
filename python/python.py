import os
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
import csv
import threading
import subprocess
import torch.nn.init as init
import requests
import gzip
import numpy as np

# Get process for metrics right after imports
process = psutil.Process(os.getpid())

# Define constants
EPOCHS = 10
BATCH_SIZE = 1024

# --- Custom Data Loading ---
MNIST_URLS = [
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "data/train-images-idx3-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "data/train-labels-idx1-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "data/t10k-images-idx3-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "data/t10k-labels-idx1-ubyte"),
]

def download_file(url, dest_path_gz):
    print(f"Downloading {url} to {dest_path_gz}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path_gz, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    dest_path = dest_path_gz.replace('.gz', '')
    print(f"Extracting {dest_path_gz} to {dest_path}...")
    with gzip.open(dest_path_gz, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(dest_path_gz)

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        return np.frombuffer(f.read(), dtype=np.uint8)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
gpu_available_for_metrics = False
if device.type == "cuda":
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        gpu_name = result.stdout.strip()
        print(f"nvidia-smi detected GPU: {gpu_name}. GPU metrics enabled.")
        gpu_available_for_metrics = True
        torch.cuda.init()
        _ = torch.tensor([1.0, 2.0]).to(device)
        print("CUDA Initialized and allocation test successful.")
    except Exception as e:
        print(f"WARN: nvidia-smi command failed or CUDA initialization error: {e}. GPU metrics via nvidia-smi disabled.")
        gpu_available_for_metrics = False
else:
    print("No GPU found. Using CPU for computation.")
    gpu_available_for_metrics = False

# Create directories
os.makedirs('stats', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Download and load data using custom functions
for url, path in MNIST_URLS:
    if not os.path.exists(path):
        download_file(url, path + ".gz")

train_images_np = load_mnist_images('data/train-images-idx3-ubyte')
train_labels_np = load_mnist_labels('data/train-labels-idx1-ubyte')
test_images_np = load_mnist_images('data/t10k-images-idx3-ubyte')
test_labels_np = load_mnist_labels('data/t10k-labels-idx1-ubyte')

train_images = torch.from_numpy(train_images_np.copy()).float() / 255.0
train_labels = torch.from_numpy(train_labels_np.copy()).long()
test_images = torch.from_numpy(test_images_np.copy()).float() / 255.0
test_labels = torch.from_numpy(test_labels_np.copy()).long()

# Move entire dataset to the target device
print("Pre-loading dataset to device...")
train_images, train_labels = train_images.to(device), train_labels.to(device)
test_images, test_labels = test_images.to(device), test_labels.to(device)
print("Dataset pre-loaded.")

train_dataset_gpu = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset_gpu = torch.utils.data.TensorDataset(test_images, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset_gpu, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset_gpu, batch_size=BATCH_SIZE, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        if self.fc1.bias is not None: init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None: init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def collect_detailed_metrics(start_time, stop_event, detailed_metrics_data, process):
    while not stop_event.is_set():
        try:
            current_time_sec = time.time()
            elapsed_time = current_time_sec - start_time
            gpu_util, gpu_power = 0.0, 0.0
            if gpu_available_for_metrics:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,power.draw', '--format=csv,noheader,nounits', '-i', '0'],
                    capture_output=True, text=True, check=True, timeout=1
                )
                util_str, power_str = result.stdout.strip().split(',')
                gpu_util, gpu_power = float(util_str.strip()), float(power_str.strip())
            
            memory_in_mb = process.memory_info().rss / (1024 * 1024)
            detailed_metrics_data.append((elapsed_time, gpu_util, gpu_power, memory_in_mb))
            
            processing_time = time.time() - current_time_sec
            time.sleep(max(0, 0.1 - processing_time))
        except Exception:
            if not stop_event.is_set(): continue

# --- Main Execution ---
NUM_RUNS = 3
run_total_times, run_final_accuracies, run_avg_epoch_times = [], [], []
run_avg_gpu_utils, run_avg_gpu_powers, run_avg_mems, run_peak_mems = [], [], [], []

criterion = nn.CrossEntropyLoss()

for run in range(NUM_RUNS):
    print(f"\n--- Starting Run {run + 1}/{NUM_RUNS} ---")
    
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-8)
    
    run_epoch_times = []
    detailed_metrics_data = []
    stop_event = threading.Event()
    start_time = time.time()

    metrics_thread = threading.Thread(target=collect_detailed_metrics, args=(start_time, stop_event, detailed_metrics_data, process))
    metrics_thread.daemon = True
    metrics_thread.start()

    # Open epoch metrics file only on first run
    epoch_metrics_file = None
    if run == 0:
        epoch_metrics_file = open('stats/py_metrics.csv', 'w', newline='')
        epoch_writer = csv.writer(epoch_metrics_file)
        epoch_writer.writerow(['epoch', 'train_accuracy', 'valid_accuracy', 'epoch_time'])

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        train_correct, train_total = 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        valid_correct, valid_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
        valid_accuracy = 100 * valid_correct / valid_total
        
        epoch_duration = time.time() - epoch_start_time
        run_epoch_times.append(epoch_duration)
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Acc: {train_accuracy:.2f}%, Valid Acc: {valid_accuracy:.2f}%, Epoch Time: {epoch_duration:.2f}s')

        # Write to epoch metrics file on first run
        if run == 0 and epoch_metrics_file:
            epoch_writer.writerow([epoch + 1, f"{train_accuracy:.2f}", f"{valid_accuracy:.2f}", f"{epoch_duration:.2f}"])

    total_time_for_run = time.time() - start_time
    stop_event.set()
    metrics_thread.join(timeout=2)

    # Close epoch file if it was opened
    if epoch_metrics_file:
        epoch_metrics_file.close()

    run_total_times.append(total_time_for_run)
    run_final_accuracies.append(valid_accuracy)
    run_avg_epoch_times.append(mean(run_epoch_times))
    
    if detailed_metrics_data:
        run_avg_gpu_utils.append(mean([d[1] for d in detailed_metrics_data]))
        run_avg_gpu_powers.append(mean([d[2] for d in detailed_metrics_data]))
        run_avg_mems.append(mean([d[3] for d in detailed_metrics_data]))
        run_peak_mems.append(max([d[3] for d in detailed_metrics_data]))
    else:
        run_avg_gpu_utils.append(0); run_avg_gpu_powers.append(0); run_avg_mems.append(0); run_peak_mems.append(0)
    
    print(f"Run {run + 1} finished in {total_time_for_run:.2f} seconds.")

    if run == 0:
        print(f"\nCollected {len(detailed_metrics_data)} detailed metrics points (for CSV and summary)")
        if epoch_metrics_file:
            print("Epoch metrics saved to stats/py_metrics.csv")
        with open('stats/py_metrics_detailed.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'gpu_utilization_percent', 'gpu_power_watts', 'memory_mb'])
            for row in detailed_metrics_data:
                writer.writerow([f"{row[0]:.2f}", f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}"])
        print("Detailed metrics saved to stats/py_metrics_detailed.csv")

# --- Final Summary Calculation ---
avg_total_time = mean(run_total_times)
avg_final_accuracy = mean(run_final_accuracies)
avg_avg_epoch_time = mean(run_avg_epoch_times)
avg_gpu_util = mean(run_avg_gpu_utils)
avg_gpu_power = mean(run_avg_gpu_powers)
avg_mem_usage = mean(run_avg_mems)
max_peak_mem = max(run_peak_mems) if run_peak_mems else 0
energy_joules = (run_avg_gpu_powers[0] * run_total_times[0]) if run_avg_gpu_powers else 0

print("\n--- Benchmark Complete ---")
print(f"Average Total Time over {NUM_RUNS} runs: {avg_total_time:.2f}s")
print(f"Average Final Accuracy over {NUM_RUNS} runs: {avg_final_accuracy:.2f}%")
print(f"Average Epoch Time over {NUM_RUNS} runs: {avg_avg_epoch_time:.2f}s")

with open('stats/py_training_summary.txt', 'w') as f:
    f.write('Training Summary:\n')
    f.write(f'Number of Runs: {NUM_RUNS}\n')
    f.write(f'Device: {device}\n')

    for i in range(NUM_RUNS):
        f.write(f'\n--- Run {i + 1}/{NUM_RUNS} ---\n')
        f.write(f'Total Time: {run_total_times[i]:.2f} seconds\n')
        f.write(f'Final Accuracy: {run_final_accuracies[i]:.2f}%\n')
        f.write(f'Average Epoch Time: {run_avg_epoch_times[i]:.2f} seconds\n')
        if gpu_available_for_metrics:
            f.write(f'Average GPU Utilization: {run_avg_gpu_utils[i]:.1f}%\n')
            f.write(f'Average GPU Power Draw: {run_avg_gpu_powers[i]:.1f} W\n')
        f.write(f'Average Process Memory Usage: {run_avg_mems[i]:.1f} MB\n')
        f.write(f'Peak Process Memory Usage (this run): {run_peak_mems[i]:.2f} MB\n')

    f.write(f'\n--- Averages over {NUM_RUNS} runs ---\n')
    f.write(f'Average Total Time: {avg_total_time:.2f} seconds\n')
    f.write(f'Average Final Accuracy: {avg_final_accuracy:.2f}%\n')
    f.write(f'Average Epoch Time: {avg_avg_epoch_time:.2f} seconds\n')
    if gpu_available_for_metrics:
        f.write(f'Average GPU Utilization: {avg_gpu_util:.1f}%\n')
        f.write(f'Average GPU Power Draw: {avg_gpu_power:.1f} W\n')
        f.write(f'Approx Total GPU Energy Consumed (Run 1): {energy_joules:.2f} J\n')
    f.write(f'Average Process Memory Usage: {avg_mem_usage:.1f} MB\n')
    f.write(f'Overall Peak Process Memory Usage: {max_peak_mem:.2f} MB\n')
    
if device.type == "cuda":
    torch.cuda.empty_cache()

print('\nFinished Training and summary saved to stats/py_training_summary.txt.')
