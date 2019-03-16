#!/usr/bin/env bash
echo "Start"
# Exit immediately if a command exits with a non-zero status.
set -e
EXP_NAME=${1}
NUM_ITERATIONS=${2}
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

echo "EXP_NAME=${EXP_NAME}"
# Set up the working directories.
DATASET_FOLDER="msim3"
EXP_FOLDER="exp/${EXP_NAME}"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/export"
echo "INIT_FOLDER=${INIT_FOLDER}"
echo "DATASET_FOLDER=${DATASET_FOLDER}"
echo "EXP_FOLDER=${EXP_FOLDER}"
echo "TRAIN_LOGDIR=${TRAIN_LOGDIR}"
echo "EXPORT_DIR=${EXPORT_DIR}"

mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EXPORT_DIR}"


# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
#http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz

#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
TF_INIT_CKPT="deeplabv3_cityscapes_train_2018_02_06.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

#deeplabv3_pascal_trainval_2018_01_04.tar.gz
TFRECORD_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tfrecord"

# Train 10 iterations.
#NUM_ITERATIONS=40000
echo "Training"
CUDA_VISIBLE_DEVICES="1,2,3" python "${WORK_DIR}"/train.py \
  --num_clones=3 \
  --num_replicas=3 \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=12 \
  --dataset=msim3 \
  --initialize_last_layer=false \
  --last_layers_contain_logits_only=true \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TFRECORD_DIR}" \
#  --fine_tune_batch_norm=true \

