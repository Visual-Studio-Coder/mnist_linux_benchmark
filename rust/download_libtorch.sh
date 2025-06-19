#!/bin/bash

# This script downloads LibTorch and sets up the environment

# --- USER: VERIFY THESE on pytorch.org/get-started/locally/ ---
LIBTORCH_VERSION="2.6.0"
# Try cu124 as it's available and compatible with >=12.4 drivers
CUDA_VERSION_TAG="cu124" 
# --- END USER VERIFICATION ---

ABI="cxx11-abi"
BASE_FILENAME="libtorch-${ABI}-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_VERSION_TAG}"
ZIP_FILENAME_URL="${BASE_FILENAME//+/%2B}.zip" # URL uses %2B
ZIP_FILENAME_LOCAL="${BASE_FILENAME}.zip"      # Local file uses +

# Construct the final URL (adjust path if needed based on PyTorch website)
DOWNLOAD_URL="https://download.pytorch.org/libtorch/${CUDA_VERSION_TAG}/${ZIP_FILENAME_URL}"

# Clean up any existing LibTorch
echo "Removing any existing LibTorch installation..."
rm -rf ./libtorch
# Remove potentially incomplete downloads
rm -f libtorch-*.zip*

# Download LibTorch using the verified URL
echo "Downloading LibTorch ${LIBTORCH_VERSION} with ${CUDA_VERSION_TAG} support from:"
echo "${DOWNLOAD_URL}"
wget "${DOWNLOAD_URL}" -O "${ZIP_FILENAME_LOCAL}"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Download failed! Please double-check the URL and your internet connection."
    exit 1
fi

# Extract the downloaded zip using the correct local filename
echo "Extracting LibTorch from ${ZIP_FILENAME_LOCAL}..."
unzip "${ZIP_FILENAME_LOCAL}"

# Check if extraction was successful
if [ -d "./libtorch" ]; then
    echo "LibTorch successfully extracted!"
    
    # Optional: Remove the zip file to save space
    rm "${ZIP_FILENAME_LOCAL}"
    
    # Display some info about the installation
    echo "LibTorch installed at: $(pwd)/libtorch"
    echo "CUDA libraries found:"
    ls -la ./libtorch/lib/*cuda*
    
    # Set environment variables for the current shell session
    # Use absolute path for LIBTORCH
    export LIBTORCH="$(pwd)/libtorch" 
    # Prepend to LD_LIBRARY_PATH to ensure it's found first
    export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH}" 
    
    echo ""
    echo "Environment variables have been set for THIS terminal session:"
    echo "LIBTORCH=${LIBTORCH}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    echo ""
    echo "To use these in new terminals, export them manually or add to your .bashrc"
    echo ""
    
    echo "Ready to build and run the Rust benchmark!"
    echo "cd rust_benchmark"
    echo "export CUDA_VISIBLE_DEVICES=0  # Optional: explicitly select GPU 0"
    echo "cargo clean"
    echo "cargo run --release"
else
    echo "Error: LibTorch extraction failed! Check if unzip command worked and ${ZIP_FILENAME_LOCAL} is valid."
    exit 1
fi
