# HEqBM

Instructions for setting up virtual environment and installing all required packages:

1 - Create a new virtual environment with python>=3.8
    Suggested version is 3.10
    Using conda, the script to run is 'conda create -n "heqbm" python=3.10'

2 - Activate the newly created virtual environment: 'conda activate heqbm'

3 - Clone the forked NequIP repository using 'git clone https://github.com/Daniangio/nequip.git'

4 - Install PyTorch >= 1.10, <=1.13, !=1.9. PyTorch can be installed following the [instructions from their documentation](https://pytorch.org/get-started/locally/). Note that neither `torchvision` nor `torchaudio`, included in the default install command, are needed for NequIP.

4 - Go inside the NeqIP folder and run 'pip install -e .' to install nequip and all its required packages inside your virtual environment

5 - Return inside the HEqBM folder and run 'pip install -e .' to install heqbm and all its required packages inside your virtual environment