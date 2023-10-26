#!/bin/bash
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=23:00:00
#$ -N TRAINING
#$ -o eo_file_train
#$ -e eo_file_train

. /etc/profile.d/modules.sh
source /home/8/18B15885/.bashrc
python --version
pip freeze

module load cuda/11.1.1 cudnn/8.8.1 gcc/8.3.0-cuda
module list

export NCCL_IB_DISABLE=1

./training_scripts/pda_nda.sh
