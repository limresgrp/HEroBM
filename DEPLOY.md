# HEroBM Deployment Guide for Shared Systems
This guide outlines the one-time setup process to make the HEroBM software available to all users on a shared machine.
The software will be installed in `/apps/herobm`.

The chosen deployment method is a self-contained Conda environment.
This ensures that all dependencies, including Python, PyTorch, and the correct CUDA toolkit version, are isolated and will not conflict with other software on the system.

## One-Time Administrator Setup
These steps should be performed by a system administrator or a user with write access to the `/apps` directory.

## 1. Create the Installation Directory
First, create the main directory for the application.
```bash
    export HEROBM_ROOT=/apps/herobm
    sudo mkdir -p $HEROBM_ROOT
    sudo chown $USER $HEROBM_ROOT # Temporarily take ownership
```
## 2. Set Up the Conda Environment
We will create a new Conda environment inside the application directory. This keeps everything self-contained.
Note: Ensure you have a Conda installation (like Miniconda or Anaconda) available.
```bash
    cd $HEROBM_ROOT
    # Create the environment with a specific Python version
    conda create --prefix ./env python=3.10 -y
    # Activate the new environment to install packages into it
    conda activate ./env
```
## 3. Install Dependencies
Install all required packages, including the specific CUDA-enabled version of PyTorch.
```bash
    # Install packages from conda-forge first
    conda install -c conda-forge pdbfixer -y

    # Install PyTorch with a specific CUDA version
    # (Adjust cuda version e.g., 11.8 as needed for your system's drivers)
    pip install torch --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

## 4. Install Local HEroBM Packages
Clone your projects and install them in "editable" mode.
This allows you to update the code by simply pulling the latest changes from Git without needing to reinstall.
```bash
    # Clone your repositories into a 'src' directory
    mkdir -p $HEROBM_ROOT/src
    cd $HEROBM_ROOT/src

    # IMPORTANT: Replace with the actual URLs or paths to your repos
    git clone git@github.com:your-user/CGmap.git
    git clone git@github.com:your-user/GEqTrain.git
    git clone git@github.com:your-user/HEroBM.git

    # Install them in editable mode
    pip install -e ./CGmap
    pip install -e ./GEqTrain
    pip install -e ./HEroBM

    # Deactivate the environment
    conda deactivate
```

## 5. Deploy the sourceme.sh script
Copy the sourceme.sh script (provided below) into the `$HEROBM_ROOT` directory and make it executable.
```bash
    cp path/to/your/sourceme.sh $HEROBM_ROOT/sourceme.sh
    chmod +x $HEROBM_ROOT/sourceme.sh
```

## 6. (Optional) Deploy Models
Create a directory to store the deployed, pre-trained models so users can easily access them.
```bash
    mkdir -p $HEROBM_ROOT/models
    # scp your-deployed-model.pth user@host:$HEROBM_ROOT/models/
```

## 7. Finalize Permissions
Return ownership of the directory to root so that standard users cannot modify the installation.
```bash
    # Change ownership of the entire application to root
    sudo chown -R root:root $HEROBM_ROOT

    # Step 1: Set all directories to be accessible (rwxr-xr-x)
    sudo find $HEROBM_ROOT -type d -exec chmod 755 {} \;

    # Step 2: Set all regular files to be readable (rw-r--r--)
    sudo find $HEROBM_ROOT -type f -exec chmod 644 {} \;

    # Step 3: Make all scripts in the 'bin' directory executable (rwxr-xr-x)
    # This overrides the previous step for these specific files.
    if [ -d "$HEROBM_ROOT/env/bin" ]; then
        sudo chmod 755 $HEROBM_ROOT/env/bin/*
    
    # Step 4: Fix to allow writing of bead_types.yaml
    sudo chmod -R 777 "$HEROBM_ROOT/src/CGmap/cgmap/data/"
fi
```

The setup is now complete.
Instructions for End-UsersTo use the HEroBM backmapping tool, you just need to source the setup script.
This will activate the correct environment and make the herobm-backmap command available.
Usage:
```bash
    # Load the HEroBM environment
    source /apps/herobm/sourceme.sh

    # You can now run the backmapping command
    # Example:
    herobm-backmap --cg -i input.pdb -it trajectory.xtc -mo /apps/herobm/models/your-model.pth -o ./output_structures -s 'protein'

    # To leave the HEroBM environment when you are done
    conda deactivate
```

# sourceme.sh

```
#!/bin/bash

# ==============================================================================
#  sourceme.sh for HEroBM
# ==============================================================================
#
#  Purpose:
#  This script configures the user's shell environment to use the HEroBM
#  software suite. It activates the self-contained Conda environment which
#  includes all necessary dependencies like PyTorch with CUDA support.
#
#  Usage:
#  source /path/to/this/script/sourceme.sh
#
# ==============================================================================

# Get the directory where this script is located
export HEROBM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if the environment directory exists
if [ ! -d "$HEROBM_ROOT/env" ]; then
    echo "ERROR: The HEroBM Conda environment was not found at $HEROBM_ROOT/env"
    echo "Please contact the system administrator for assistance."
    return 1
fi

# Activate the conda environment
# Using 'conda activate' is the recommended way
echo "Activating HEroBM environment..."
conda activate "$HEROBM_ROOT/env"

# Add the location of the deployed models to an environment variable
# so users can easily reference it.
export HEROBM_MODELS_DIR="$HEROBM_ROOT/models"

echo "=========================================================="
echo " HEroBM Environment is now active."
echo ""
echo "  - Root Directory: $HEROBM_ROOT"
echo "  - Deployed Models: $HEROBM_MODELS_DIR"
echo ""
echo "  You can now use the 'herobm-backmap' command."
echo "  To deactivate this environment, run: conda deactivate"
echo "=========================================================="
```