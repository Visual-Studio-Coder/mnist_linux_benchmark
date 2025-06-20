# Rust MNIST Benchmark

## What It Does

This program uses the `tch-rs` crate (a Rust wrapper for LibTorch) to train a simple neural network on the MNIST dataset. It performs the following steps:
1.  Downloads the MNIST dataset.
2.  Loads the entire dataset into GPU memory.
3.  Trains a model for 10 epochs.
4.  Collects and saves performance stats (time, accuracy, GPU usage, memory) to the `stats/` directory in the project root.

## How to Run

### Prerequisites
- An NVIDIA GPU with CUDA.
- The Rust toolchain (`cargo`).
- A downloaded copy of **LibTorch**. You must set the `LIBTORCH` environment variable to point to the extracted LibTorch directory.

### Run Command
From the `rust/` directory, run:
```bash
# Set this to your LibTorch path
export LIBTORCH=/path/to/your/libtorch

# Run the benchmark in release mode for performance
cd rust_benchmark
cargo run --release
```