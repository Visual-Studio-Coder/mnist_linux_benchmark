# Python

## Setup

### 1. Install Anaconda or Miniconda
If you don't have Anaconda or Miniconda installed, you can download and install it from the following links:
- [Anaconda](https://www.anaconda.com/products/distribution)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create and activate a Conda environment
Open your terminal and run the following commands to create and activate a new Conda environment:

```bash
conda create --name mnist_env python=3.11 -y
conda activate mnist_env
```

### 3. Install required Python packages
**With the `mnist_env` environment active**, install the packages:

```bash
# Check driver first
nvidia-smi 

# Install PyTorch with matching CUDA support (e.g., cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies (NO pynvml needed now)
pip install psutil matplotlib pandas numpy 
```

Verify installation **within the active environment**:
```bash
# Verify PyTorch CUDA
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
# Verify nvidia-smi command works (used for GPU metrics)
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
```
If `GPU available` is `False` or `nvidia-smi` fails, check the Troubleshooting section.

### 4. Run the `python.py` script
Navigate to the directory where `python.py` is located and run the script:

```bash
cd /Users/vaibhavsatishkumar/mnist/python
python python.py
```

This will start the training process using the Multilayer Perceptron (MLP) defined in the script. The model will automatically use GPU if available, otherwise it will fallback to CPU.

### 5. Run the `collect_metrics.py` script (if needed separately)
If you need to run the `collect_metrics.py` script separately, you can do so by running:

```bash
python collect_metrics.py <list of process-names> <collection-interval-sec> <output-filepath>
```

For example, to monitor the CPU and memory consumption of the `python` process every 5 seconds and output to the console:

```bash
python collect_metrics.py python 5
```

To output the metrics to a file:

```bash
python collect_metrics.py python 5 metrics.log
```

## Troubleshooting

### GPU Issues
- **`UserWarning: CUDA initialization: CUDA unknown error... Setting the available devices to be zero.`** or **`RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable`**: This indicates PyTorch cannot properly initialize or communicate with your GPU driver.
    1.  **Compatibility Check:** Verify NVIDIA driver, CUDA toolkit (often bundled), and PyTorch CUDA version (`cuXXX`) compatibility using `nvidia-smi` and the PyTorch installation guide.
    2.  **Minimal Test:** Run a simple script (`test_cuda.py` example below) to isolate the issue.
        ```python
        # test_cuda.py
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"Device Name: {torch.cuda.get_device_name(0)}")
                a = torch.tensor([1.0, 2.0]).cuda()
                print(f"Success! Tensor on CUDA: {a}")
            except Exception as e:
                print(f"CUDA Error: {e}")
        ```
    3.  **Permissions:** Check Linux device permissions. Run `ls -l /dev/nvidia*` and `groups $(whoami)`. Ensure your user is in the group that owns `/dev/nvidia*` (often `video`). If not, add yourself (`sudo usermod -aG video $USER`) and **log out/log back in**.
    4.  **Environment Variables:** Ensure `CUDA_VISIBLE_DEVICES` is unset or correct *before* starting Python (`unset CUDA_VISIBLE_DEVICES`).
    5.  **Clean Driver Reinstall:** This is often necessary. Completely remove existing NVIDIA drivers, reboot, install the latest official driver for your GPU, and reboot again. Then test.
    6.  **Fresh Conda Environment:** Create a new environment (`conda create --name test_env python=3.11 -y; conda activate test_env`) and reinstall PyTorch and dependencies there.
    7.  **Check `nvidia-smi`:** If `nvidia-smi` itself fails or shows errors, the driver installation is definitely the problem.
    8.  **Reboot:** Sometimes a simple reboot helps clear transient issues.

### Package Installation Issues
- If you have issues with pip, you can try installing with conda:
  ```bash
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
- For specific errors, check the PyTorch documentation for your platform.

## Additional Information
- The script automatically detects whether a GPU is available and uses it if present.
- **GPU utilization and power metrics** are collected by calling the `nvidia-smi` command-line tool in a background thread if a GPU is used. Ensure `nvidia-smi` is in your system PATH.
- Process memory usage (RSS) is tracked using `psutil`.
- Training progress, metrics, and plots are saved in the `stats` directory.
- **GPU energy consumption** is approximated by averaging the power draw reported by `nvidia-smi` over the script's runtime.
- CPU metrics are no longer the primary focus but could be re-added if needed.
- Make sure you have the necessary Python version by running:
  ```bash
  python --version
  ```
- If you don't have Python installed, you can download it from [here](https://www.python.org/downloads/).
