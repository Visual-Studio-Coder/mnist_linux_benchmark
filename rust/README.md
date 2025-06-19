# Rust MNIST Benchmark with tch-rs

This directory contains a Rust implementation for training a simple MLP on the MNIST dataset using the `tch-rs` crate (LibTorch bindings). It collects performance and resource metrics.

## Setup

### 1. Install Rust
If you don't have Rust installed, follow the instructions at [https://rustup.rs/](https://rustup.rs/).

### 2. Install System Dependencies (Linux)
You might need standard build tools and potentially `cmake`:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config libssl-dev # Example for Debian/Ubuntu
```

### 3. Download LibTorch (CPU or GPU)

**Crucial Step:** You need the LibTorch library. Choose **one** option below.

*   **Option A: CPU-only:**
    1.  Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and find the Linux/LibTorch/C++/CPU download URL if the one below is outdated.
    2.  Download using `wget` (replace URL if necessary):
        ```bash
        # Example URL for LibTorch 2.6.0 CPU - Verify on PyTorch website
        wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip
        ```
    3.  Unzip the downloaded file:
        ```bash
        unzip libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip 
        ```
    4.  Move the extracted `libtorch` folder into the `rust/` directory:
        ```bash
        # Ensure no old libtorch exists first: rm -rf ./libtorch 
        mv libtorch ./
        ```
        *(You should now have `rust/libtorch/lib`, `rust/libtorch/include`, etc.)*

*   **Option B: GPU (CUDA):**
    1.  Ensure you have a compatible NVIDIA driver installed (`nvidia-smi` runs).
    2.  Verify the CUDA version supported by your driver. The URL below is for **CUDA 11.8**. If you need a different version (like 12.1), get the correct URL from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) (Select Linux, LibTorch, C++/Java, CUDA X.Y).
    3.  Download using `wget` (using the URL you provided for cu118):
        ```bash
        wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu118.zip
        ```
    4.  Unzip the downloaded file:
        ```bash
        unzip libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu118.zip
        ```
    5.  Move the extracted `libtorch` folder into the `rust/` directory, replacing any existing one:
        ```bash
        # Ensure no old libtorch exists first: rm -rf ./libtorch 
        mv libtorch ./
        ```
        *(You should now have `rust/libtorch/lib`, `rust/libtorch/include`, etc.)*

### 4. Configure the Build (Automatic)
The project is configured to automatically find the `libtorch` library located in `rust/libtorch`. This is handled by the `.cargo/config.toml` file inside the `rust_benchmark` directory.

No manual environment variable setup (like `LIBTORCH` or `LD_LIBRARY_PATH`) is required for building or running.

## Building and Running

1.  Navigate to the `rust_benchmark` directory:
    ```bash
    cd /path/to/your/mnist_linux-master/rust/rust_benchmark
    ```
2.  Build the project (this will download dependencies like `tch-rs`):
    ```bash
    cargo build --release
    ```
    *(The first build might take a while)*
3.  Run the benchmark:
    ```bash
    ./target/release/rust_benchmark
    ```

The script will:
*   Download the MNIST dataset to `rust_benchmark/data` if not present.
*   Train the MLP for 10 epochs.
*   Use CUDA if available and the correct LibTorch version is present.
*   Collect metrics in a background thread.
*   Save results to the `rust_benchmark/stats` directory:
    *   `metrics.csv`: Per-epoch summary (accuracy, time).
    *   `metrics_detailed.csv`: Fine-grained metrics (time, GPU util/power, memory).
    *   `training_summary.txt`: Overall summary text.

## Metrics Collected
- **GPU utilization (%) and power (W)**: Collected by calling `nvidia-smi` if a GPU is used and the command is available.
- **Process memory usage (RSS, MB)**: Collected using the `sysinfo` crate.
- **Epoch Time (s)**
- **Train/Validation Accuracy (%)**
- **Total Runtime (s)**
- **Approximate Total GPU Energy Consumed (J)**: Calculated from average power over time.

## Troubleshooting
- **Build Errors like `torch/torch.h: No such file or directory`:** This is the most common issue.
    1.  **Clean the build cache.** This is critical. Before trying anything else, run the following command from inside the `rust_benchmark` directory:
        ```bash
        cargo clean
        ```
    2.  **Verify `libtorch` location.** Ensure the extracted `libtorch` directory is located at `rust/libtorch`.
    3.  **Verify config paths.** The file `rust/rust_benchmark/.cargo/config.toml` tells the build where to find `libtorch`. Make sure the paths inside are correct for your system. If you move the project, you must update these paths.
- **Runtime Errors like `cannot open shared object file`:** The project is configured to avoid this, but if it happens, it means the `rpath` linker setting failed. Ensure you have `build-essential` installed and try a `cargo clean` followed by a `cargo build --release`.
- **GPU Not Used:** Verify you downloaded and are using the **CUDA-enabled** version of LibTorch. Check that `nvidia-smi` works and your driver is compatible. The script output should show `Using device: Cuda(0)`.
- **`nvidia-smi` Errors:** If the script prints warnings about `nvidia-smi`, ensure the command works from your terminal and is in the system PATH. GPU metrics will be disabled otherwise.

## Additional Setup for Plotting
The benchmark calls a Python script (`plot_metrics.py`) to generate graphs from the collected data.

1.  **Install Python:** Ensure you have Python 3 installed. Check with `python --version` or `python3 --version`. Download from [python.org](https://www.python.org/downloads/) if needed.
2.  **Install Python Packages:** The plotting script requires `matplotlib` and `pandas`. Install them using pip:
    ```bash
    pip install matplotlib pandas
    # or use pip3 if needed:
    # pip3 install matplotlib pandas
    ```

Feb 14
setup linux env
download vs code
download rust and python
plan to try rust
met torch-sys issue
### Feb16 
- solved torch sys issue by restalling lib, update env variebles, transfer all the dll file from /lib to /bin because Libtorch is a static packege
- Rewsearch about collecting data not by every epoche but by every 0.1 sec
- Research about collecting data of the specific Process using PID to track
- solve some grammar issue and successfully collected datas and graph it
- future plan:
1. energy consumption
2. may use GPU
3. Apply the same method on other languages