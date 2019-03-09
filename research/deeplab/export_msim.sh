
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

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path=/home/hannes/deeplab/datasets/msim3/exp/my_exp/train/model.ckpt-100 \
  --export_path="${WORK_DIR}/frozen_inference_graph.pb" \
  --model_variant="mobilenet_v2" \
  --num_classes=34 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0 \
  --max_number_of_iterations=1