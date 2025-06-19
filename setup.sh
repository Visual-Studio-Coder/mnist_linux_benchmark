#!/usr/bin/env bash

# Function to check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Conda could not be found. Please install Anaconda or Miniconda."
        exit 1
    fi
}

# Check if conda is installed
check_conda

# Create and activate a new conda environment
ENV_NAME="mnist"

# Remove existing conda environment if it exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Removing existing conda environment..."
    conda remove --name "$ENV_NAME" --all -y
fi

# Create new conda environment with Python 3.11
echo "Creating new conda environment..."
conda create --name "$ENV_NAME" python=3.11 -y

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install required packages
echo "Installing required packages..."
conda install numpy=1.24.3 matplotlib=3.7.1 -y

# Check if LIBTORCH is set, if not prompt the user
if [ -z "$LIBTORCH" ]; then
    read -p "Please enter the path to libtorch: " LIBTORCH
    export LIBTORCH
fi

# Set PyTorch environment variables
export DYLD_LIBRARY_PATH="${LIBTORCH}/lib:${DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH}"
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Ensure the virtual environment is isolated
export PYTHONPATH=$(conda info --base)/envs/$ENV_NAME/lib/python3.11/site-packages

# Run the plotting script
echo "Running plotting script..."
python plot_metrics.py

# Deactivate conda environment
conda deactivate
