# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=slurmout/sample_%j.out
#SBATCH --error=slurmout/sample_%j.err
#SBATCH --mem=32GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --constraint=volta32gb


# Start clean
#module purge

# Load what we need
#module load anaconda3
source activate in-context-learning
#/private/home/kartikahuja/.conda/envs/in-context-learning
conda info --envs


#


python iclmlp_cln_jmtd.py --config confs/linear_regression_icl2.yaml

