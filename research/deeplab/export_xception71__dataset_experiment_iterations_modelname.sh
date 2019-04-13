#!/usr/bin/env bash


# Args
DATASET_NAME=${1}
EXP_NAME=${2}
NUM_ITERATIONS=${3}
MODEL_NAME=${4}
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
#DATASET_NAME="aug_msim"
EXP_FOLDER="exp/${EXP_NAME}"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_NAME}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_NAME}/${EXP_FOLDER}/train"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_NAME}/${EXP_FOLDER}/export"

# Export the trained checkpoint.
mkdir -p ${EXPORT_DIR}
echo $NUM_ITERATIONS
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_71" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=34 \
  --crop_size=1281 \
  --crop_size=721 \
  --inference_scales=1.0 \
  --max_number_of_iterations=1 \

# Compress the graph
tar -cvzf "${EXPORT_DIR}"/${MODEL_NAME}.tar.gz "${EXPORT_PATH}"
echo "Model saved to ${EXPORT_DIR}/${MODEL_NAME}.tar.gz"
# Remove the uncompressed graph
rm "${EXPORT_PATH}"
