# HEroBM

Instructions for setting up virtual environment and installing all required packages:

1 - Create a new virtual environment with python>=3.8
    Suggested version is 3.10
    Using conda, the script to run is 'conda create -n "herobm" python=3.10'

2 - Activate the newly created virtual environment: 'conda activate herobm'

3 - Clone the GEqTrain repository using 'git clone https://github.com/limresgrp/GEqTrain.git'

4 - Install PyTorch >= 1.10, <=1.13, !=1.9. PyTorch can be installed following the [instructions from their documentation](https://pytorch.org/get-started/locally/). Note that neither `torchvision` nor `torchaudio`, included in the default install command, are needed for GEqTrain.

5 - Go inside the GEqTrain folder and run 'pip install -e .' to install geqtrain and all its required packages inside your virtual environment

6 - Clone the CGMap repository using 'git clone https://github.com/limresgrp/CGmap.git'

7 - Go inside the CGMap folder and run 'pip install -e .' to install cgmap and all its required packages inside your virtual environment

8 - Return inside the HEroBM folder and run 'pip install -e .' to install herobm and all its required packages inside your virtual environment

--- Soon GEqTrain and CGMap repositories will be provided as a Conda package, greatly simplifying the installation process ---