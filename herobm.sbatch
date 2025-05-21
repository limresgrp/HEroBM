#!/bin/bash
#SBATCH --job-name=herobm-train
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --environment=geqtrain
#SBATCH --account=u8
#SBATCH --partition=normal
#SBATCH --mem=64GB


cd $HOME/HEroBM/
source ./herobm-venv/bin/activate

set -x

# Use the first command-line argument as the CONFIG variable
CONFIG=$1

# Check if CONFIG is provided
if [ -z "$CONFIG" ]; then
  echo "Error: CONFIG argument is missing."
  exit 1
fi

geqtrain-train $CONFIG -ws 4