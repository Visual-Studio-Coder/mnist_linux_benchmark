# Python MNIST Benchmark

## What It Does

This script uses PyTorch to train a simple neural network on the MNIST dataset. It performs the following steps:
1.  Downloads the MNIST dataset.
2.  Loads the entire dataset into GPU memory.
3.  Trains a model for 10 epochs.
4.  Collects and saves performance stats (time, accuracy, GPU usage, memory) to the `stats/` directory in the project root.

## How to Run

### Prerequisites
- An NVIDIA GPU with CUDA.
- Python and Pip installed.
- Install dependencies:
  ```bash
  pip install torch psutil requests numpy
  ```

### Run Command
From the `python/` directory, run:
```bash
python python.py
```
