#!/bin/bash

SEED=0
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/PDBbind-v2020
EXE_DIR=${ROOT_DIR}/src
EXPERIMENT_NAME=FPBR/pda_nda/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

date
python -u ${EXE_DIR}/exe/train.py \
  experiment_name=${EXPERIMENT_NAME} \
  data=[messi/pda,messi/docking,messi/cross,messi/random] \
  data.pda.root_data_dir=${DATA_DIR}/pda \
  data.pda.key_dir=${EXE_DIR}/keys/train/FPBR/pda \
  data.docking.root_data_dir=${DATA_DIR}/docking \
  data.docking.key_dir=${EXE_DIR}/keys/train/FPBR/docking \
  data.cross.root_data_dir=${DATA_DIR}/cross \
  data.cross.key_dir=${EXE_DIR}/keys/train/FPBR/cross \
  data.random.root_data_dir=${DATA_DIR}/random \
  data.random.key_dir=${EXE_DIR}/keys/train/FPBR/random \
  model=pignet_morse \
  model.short_range_A=2.1 \
  run.dropout_rate=0.1 \
  run.lr=4e-4 \
  run.batch_size=64 \
  run.save_every=1 \
  run.num_epochs=5000 \
  run.num_workers=4 \
  run.pin_memory=false \
  run.seed=${SEED}

date
