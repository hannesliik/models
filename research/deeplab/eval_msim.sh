

# Set up the working environment.
# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
cd "${CURRENT_DIR}"
DATASET_DIR="datasets"
MSIM3_FOLDER="msim3"
EXP_FOLDER="exp/my_exp"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/${EXP_FOLDER}/export"
MSIM3_DATASET="${WORK_DIR}/${DATASET_DIR}/${MSIM3_FOLDER}/tfrecord"


python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="test" \
  --model_variant="mobilenet_v2" \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --output_stride=16 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${MSIM3_DATASET}" \
  --max_number_of_evaluations=0 \
  --dataset msim3