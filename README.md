# MNIST Benchmark (Python, C++, Rust)

This project benchmarks a simple MLP model on the MNIST dataset using Python (PyTorch), C++ (LibTorch), and Rust (tch-rs).

All three benchmarks now use an identical methodology for a fair comparison:
- The entire dataset is pre-loaded onto the GPU.
- Training is performed with manual batching using shuffled indices (`torch.randperm`).
- Evaluation is performed on the entire test set at once.
- The model, optimizer, and hyperparameters are identical.

## Running the Benchmarks

### Prerequisites
- For Python: `torch`, `psutil`, `requests`, `numpy`
- For C++/Rust: A downloaded version of LibTorch. Ensure the `LIBTORCH` environment variable points to its location.
- For C++: `cmake` and a C++ compiler.
- For Rust: The Rust toolchain (`cargo`).

### Python
Navigate to the `python` directory and run:
```bash
python python.py
```

### C++
Navigate to the `cpp/cpp_benchmark` directory. The project is configured to build in `Release` mode for optimal performance.
```bash
# Ensure LIBTORCH is set, e.g.: export LIBTORCH=/path/to/libtorch
mkdir -p build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$LIBTORCH
make
./mnist_benchmark_cpp
```

### Rust

**Important:** For a fair performance comparison, you **must** run the Rust benchmark in release mode to enable compiler optimizations.

Navigate to the `rust/rust_benchmark` directory and run:
```bash
# Ensure LIBTORCH is set, e.g.: export LIBTORCH=/path/to/libtorch
cargo run --release
```

Running with `cargo run` without the `--release` flag will use a debug build, which will be significantly slower and is not suitable for performance measurement.

## Performance Notes

After aligning the core logic, data handling, and build configurations, the expected performance hierarchy is generally **C++ ≥ Python > Rust**. This might be surprising, but it can be explained by the nature of the bindings:

- **C++**: Serves as the baseline, calling the LibTorch library directly with zero abstraction overhead.
- **Python**: While an interpreted language, the expensive tensor operations are delegated to the same underlying C++ backend. The interpreter overhead is minimal for this workload, resulting in performance that is highly competitive with native C++.
- **Rust**: The `tch-rs` crate is a safe wrapper around the C++ LibTorch API. Each call from Rust to LibTorch must cross a Foreign Function Interface (FFI) boundary. While this is very fast, the accumulated overhead from thousands of calls in the training loop can make it slightly slower than the direct C++ or streamlined Python implementations.
