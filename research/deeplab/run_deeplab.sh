#!/usr/bin/env bash
python deeplab_run.py datasets/msim3/exp/deeplab_xception/export/model.tar.gz --input_dir datasets/msim3/test/image/ --output_dir datasets/msim3/test/pred_deeplab
#python deeplab_run.py datasets/msim3/exp/deeplab_xception/export/model.tar.gz --input_dir datasets/msim3/train/image/ --output_dir datasets/msim3/train/pred_deeplab
