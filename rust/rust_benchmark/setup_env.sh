#!/bin/bash

# Create conda environment
conda create -n mnist python=3.8 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mnist

# Install required packages
conda install -y pandas matplotlib seaborn

# Deactivate environment
conda deactivate
