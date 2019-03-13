#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012 using MobileNet-v2.
# Users could also modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test_mobilenetv2.sh
#
#
echo "Start"
# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py -v

DATASET_DIR="datasets"


# Set up the working directories.
MSIM3_FOLDER="msim3"
EXP_FOLDER="exp/my_exp"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/export"
echo "1"
mkdir -p "${INIT_FOLDER}"
echo "2"
mkdir -p "${TRAIN_LOGDIR}"
echo "3"
mkdir -p "${EVAL_LOGDIR}"
echo "4"
mkdir -p "${VIS_LOGDIR}"
echo "5"
mkdir -p "${EXPORT_DIR}"
echo "6"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
CKPT_NAME="deeplabv3_mnv2_pascal_train_aug"
TF_INIT_CKPT="${CKPT_NAME}_2018_01_29.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

MSIM3_DATASET="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/tfrecord"

# Train 10 iterations.
NUM_ITERATIONS=10000
echo "Training"
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=8 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=8 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/model.ckpt-30000" \
  --fine_tune_batch_norm=true \
  --initialize_last_layer=false \
  --last_layers_contain_logits_only=true \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${MSIM3_DATASET}" \
  --dataset=msim3 \
  --save_summaries_images=true
echo "Done training"

