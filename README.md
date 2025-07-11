# Unlocking Efficiency: A Multi-Language Benchmarking Study on MNIST

This repository contains the source code for the paper "Unlocking Efficiency: A Multi-Language Benchmarking Study on MNIST".

Authored by [Vaibhav Satishkumar](https://github.com/Visual-Studio-Coder) and [Yicheng "Billy" Lu](https://github.com/billylu24/).

## Description

This project provides a rigorous comparative analysis of C++, Rust, and Python for a foundational deep learning task: training a Multi-Layer Perceptron (MLP) on the MNIST dataset. The primary goal is to measure and compare the performance and resource efficiency of these distinct language environments, focusing on metrics such as execution time, memory usage, and energy consumption.

The study's methodology is designed to isolate the performance overhead of the language and its library bindings by standardizing on the PyTorch/LibTorch backend across all three implementations. This makes it a valuable contribution to the field of "Green AI" and sustainable computing.

## Dataset Information

The benchmark uses the standard MNIST dataset. The scripts are configured to automatically download the dataset from the Open Source Computer Science (OSSCI) repository into a local `data/` directory upon first execution if it is not already present.

-   **Source:** `https://ossci-datasets.s3.amazonaws.com/mnist/`

## Code Information

The repository is organized by language into three main directories. Each directory contains a specific `README.md` with detailed setup and execution instructions.

-   `/cpp`: Contains the C++ source code and build instructions.
-   `/python`: Contains the Python script and dependency information.
-   `/rust`: Contains the Rust source code and Cargo project.

All benchmark results, including detailed time-series data and summary files, are saved to a `/stats` directory, which is created automatically in the project root if it doesn't exist.

## General Requirements

The following are required to run any of the benchmarks:

-   An NVIDIA GPU with up-to-date CUDA drivers.
-   The `nvidia-smi` command-line tool must be available in the system's PATH.
-   The LibTorch library distribution must be downloaded and accessible.

## Usage Instructions

For detailed, language-specific prerequisites and step-by-step instructions on how to compile and run each benchmark, please refer to the `README.md` file inside each respective directory:

-   **[Python Instructions](python/README.md)**
-   **[C++ Instructions](cpp/README.md)**
-   **[Rust Instructions](rust/README.md)**

## Methodology

The methodology for data collection, model architecture, and hyperparameter tuning is described in detail in the main paper. A key methodological choice to ensure a purely computationally-bound benchmark was to **pre-load the entire dataset into GPU memory before timing the main training loop**. This eliminates data transfer (I/O) overhead from the performance measurements, allowing for a more direct comparison of the language environments themselves.

## Citations

If you use this code or reference our results in your research, please cite our paper.

> [Citation will be added here upon formal publication in PeerJ Computer Science]

## License

This project is licensed under the MIT License.
