# Setting Up a Virtual Environment with PyTorch and CUDA Support on CSCS

This guide explains how to set up a virtual environment with PyTorch and CUDA support on CSCS systems.

"""
This function/module is implemented based on the guidelines and best practices 
outlined in the CSCS Knowledge Base article on LLM Inference, which can be 
accessed at the following link: 
https://confluence.cscs.ch/spaces/KB/pages/852176415/LLM+Inference

Refer to the article for detailed explanations.
"""

---

## 1. Create a Dockerfile

1. Navigate to your scratch directory and create a new folder:
    ```bash
    cd $SCRATCH
    mkdir pytorch-24.01-py3-venv && cd pytorch-24.01-py3-venv
    ```

2. Create a new file named `Dockerfile` with the following content:
    ```dockerfile
    FROM nvcr.io/nvidia/pytorch:24.01-py3
    ENV DEBIAN_FRONTEND=noninteractive
    RUN apt-get update && apt-get install -y python3.10-venv && apt-get clean && rm -rf /var/lib/apt/lists/*
    ```

---

## 2. Build a Container Using Podman

1. Create a configuration file for Podman at `$HOME/.config/containers/storage.conf` with the following content:
    ```ini
    [storage]
      driver = "overlay"
      runroot = "/dev/shm/$USER/runroot"
      graphroot = "/dev/shm/$USER/root"

    [storage.options.overlay]
      mount_program = "/usr/bin/fuse-overlayfs-1.13"
    ```

2. Request a compute node from Slurm to build the container:
    ```bash
    srun --pty bash
    ```

3. Build the container using Podman:
    ```bash
    podman build -t pytorch:24.01-py3-venv .
    ```

4. Export the container as a squash file:
    ```bash
    enroot import -x mount -o pytorch-24.01-py3-venv.sqsh podman://pytorch:24.01-py3-venv
    ```

5. Exit the Slurm allocation:
    ```bash
    exit
    ```

6. Verify that the squash file (`pytorch-24.01-py3-venv.sqsh`) is created next to your `Dockerfile`. This file is a compressed container image that can be run directly by the container engine.

---

## 3. Set Up an Environment Definition File (EDF)

1. Create a file at `~/.edf/geqtrain.toml` with the following content:
    ```toml
    image = "/capstor/scratch/cscs/<user>/containers/pytorch-24.01-py3-venv/pytorch-24.01-py3-venv.sqsh"

    mounts = ["/capstor", "/users"]

    writable = true

    [annotations]
    com.hooks.aws_ofi_nccl.enabled = "true"
    com.hooks.aws_ofi_nccl.variant = "cuda12"

    [env]
    FI_CXI_DISABLE_HOST_REGISTER = "1"
    FI_MR_CACHE_MONITOR = "userfaultfd"
    NCCL_DEBUG = "INFO"
    ```

    Replace `<user>` with your actual CSCS username.

---

## 4. Clone the GEqTrain Repository

1. Clone the repository:
    ```bash
    cd $HOME
    git clone https://github.com/limresgrp/HEroBM.git
    cd GEqTrain
    ```

---

## 5. Set Up the Python Virtual Environment

1. Start the container:
    ```bash
    srun --environment=geqtrain --container-workdir=$PWD --pty bash
    ```

2. Verify that PyTorch is installed:
    ```bash
    python -m pip list | grep torch
    ```

3. Create and activate the Python virtual environment:
    ```bash
    python -m venv --system-site-packages ./geqtrain-venv
    source ./herobm-venv/bin/activate
    ```

4. Install `torch-scatter`:
    ```bash
    TORCH_CUDA_ARCH_LIST=9.0 pip install torch-scatter
    ```

5. Test the installation:
    ```bash
    python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); from torch_scatter import scatter_max'
    ```

6. Install the GEqTrain package in editable mode:
    ```bash
    pip install -e .
    ```
---

You are now ready to use the virtual environment with PyTorch and CUDA support on CSCS!