#!/bin/bash

set -e  # Exit on error

# === CONFIGURATION ===
CONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh  # Change for macOS if needed
INSTALL_DIR=$HOME/miniconda
ENV_NAME=lmao
PYTHON_VERSION=3.12

apt update
apt install portaudio19-dev libsox-dev ffmpeg


# === DOWNLOAD MINICONDA ===
echo "[1/5] Downloading Miniconda..."
wget https://repo.anaconda.com/miniconda/$CONDA_INSTALLER -O $CONDA_INSTALLER

# === INSTALL MINICONDA SILENTLY ===
echo "[2/5] Installing Miniconda to $INSTALL_DIR..."
bash $CONDA_INSTALLER -b -p $INSTALL_DIR

# === INITIALIZE CONDA ===
echo "[3/5] Initializing Conda..."
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"

# === CREATE ENVIRONMENT ===
echo "[4/5] Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create --name $ENV_NAME python=$PYTHON_VERSION

conda activate $ENV_NAME

pip install -e .

huggingface-cli login --token $HF_TOKEN

huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

source ~/miniconda/etc/profile.d/conda.sh
