# C++ MNIST Benchmark

## Prerequisites
- A C++17 compiler
- CMake (3.10+)
- [LibTorch](https://pytorch.org/)
- libcurl
- (Optional) CUDA Toolkit if you want GPU acceleration

## Setup
1. Install dependencies and set up environment variables:
   ```
   cd libtorch
   bash setup.sh
   ```
2. Configure the project:
   ```
   cd ../cpp_benchmark
   mkdir build && cd build
   cmake ..
   ```
3. Build:
   ```
   make
   ```

## Running
```
./mnist_benchmark
```
- Downloads MNIST if missing
- Trains for 10 epochs
- Saves metrics in `stats` folder
- Saves the model to `models`

## Implementation
- Uses an MLP with two `torch::nn::Linear` layers.
- Tracks training/validation accuracies, memory usage, CPU usage.
- Plots metrics via `plot_metrics.py`.

## Plotting Metrics
Use Python 3 with matplotlib and pandas to visualize metrics:
```bash
pip install matplotlib pandas
python plot_metrics.py
```
