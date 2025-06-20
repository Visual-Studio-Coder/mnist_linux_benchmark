# C++ MNIST Benchmark

## What It Does

This program uses LibTorch (the C++ version of PyTorch) to train a simple neural network on the MNIST dataset. It performs the following steps:
1.  Downloads the MNIST dataset.
2.  Loads the entire dataset into GPU memory.
3.  Trains a model for 10 epochs.
4.  Collects and saves performance stats (time, accuracy, GPU usage, memory) to the `stats/` directory in the project root.

## How to Run

### Prerequisites
- An NVIDIA GPU with CUDA.
- A C++ compiler, `cmake`, and `libcurl`.
- A downloaded copy of **LibTorch**. You must set the `LIBTORCH` environment variable to point to the extracted LibTorch directory.

### Run Commands
From the `cpp/` directory, run:
```bash
# Set this to your LibTorch path
export LIBTORCH=/path/to/your/libtorch

# Build the project
cd cpp_benchmark
mkdir -p build
cd build
cmake ..
make

# Run the benchmark
./mnist_benchmark_cpp
```
