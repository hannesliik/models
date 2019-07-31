import argparse
import tarfile
import os

import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
from tensorflow.io import gfile

parser = argparse.ArgumentParser()
parser.add_argument('input_graph')
parser.add_argument('output_graph')
args = parser.parse_args()
OUTPUT_NAME = ["SemanticPredictions"]

# read Tensorflow frozen graph
#with gfile.GFile(args.input_graph, 'rb') as tf_model:
#   tf_graphf = tensorflow.GraphDef()
#   tf_graphf.ParseFromString(tf_model.read())

tar_file = tarfile.open(args.input_graph)
for tar_info in tar_file.getmembers():
    if 'frozen_inference_graph' in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

tar_file.close()
# convert (optimize) frozen model to TensorRT model
trt_graph = trt.create_inference_graph(input_graph_def=graph_def, outputs=OUTPUT_NAME, max_batch_size=1, max_workspace_size_bytes=3 * (10 ** 9), precision_mode="FP16", is_dynamic_op=True)

# write the TensorRT model to be used later for inference
with gfile.GFile(args.output_graph, 'wb') as f:
   f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")
