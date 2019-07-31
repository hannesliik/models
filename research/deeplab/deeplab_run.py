import argparse
import os
import tarfile

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.io import gfile

# @title Helper methods
class DeepLab(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 721  # This needs to match the exported model crop size
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    CITYSCAPES_COLORMAP = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ], dtype=np.uint8)

    MILREM_COLORMAP = np.array([
        [0, 0, 0],
        [249, 12, 190],
        [85, 85, 85],
        [128, 64, 128],
        [153, 204, 102],
        [255, 153, 0],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 0, 230],
        [119, 11, 32],
        [193, 58, 58],
        [173, 10, 10],
        [22, 73, 38],
        [10, 199, 252],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [180, 165, 180],
        [150, 100, 100],
        [150, 120, 90],
        [142, 114, 68],
        [193, 196, 47],
        [31, 70, 107],
        [99, 4, 96],
        [0, 255, 0],
        [51, 255, 255],
        [52, 251, 152],
        [0, 114, 54],
        [96, 57, 19],
        [224, 143, 62],
        [45, 216, 199],
        [0, 74, 128]
    ])
    COLORMAPS = {
        'cityscapes': CITYSCAPES_COLORMAP,
        'milrem': MILREM_COLORMAP,
    }

    def __init__(self, tarball_path, colormap='cityscapes'):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        self.colormap = colormap
        graph_def = None
        if tarball_path.endswith('trt'):
            with gfile.GFile(tarball_path, 'rb') as trt_model:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(trt_model.read())
        else:
            # Extract frozen graph from tar archive.
            tar_file = tarfile.open(tarball_path)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.GraphDef.FromString(file_handle.read())
                    break

            tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)


    def run(self, image):
        """Runs inference on a single image.
        Args:
          image: A numpy array of image.

        Returns:
          seg_map: Segmentation map of `resized_image`.
        """
        # Deeplab has requirements to the image shape. Do what is necessary.
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        image = image.resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})
        seg_map = batch_seg_map[0]
        return seg_map

    def predict_image(self, image_arr):
        seg_map = self.run(image_arr)
        if seg_map.ndim != 2:
            raise ValueError('Expect 2-D input label')
        if np.max(seg_map) >= len(DeepLab.COLORMAPS[self.colormap]):
            raise ValueError('label value too large.')
        return DeepLab.COLORMAPS[self.colormap][seg_map].astype(np.uint8)

    def predict_classes(self, image_arr):
        seg_map = self.run(image_arr)
        return seg_map.astype(np.uint8)

    def predict_human_mask(self, image_arr):
        seg_map = self.run(image_arr)
        label_image = np.zeros(seg_map.shape[:2], np.uint8)
        # human class is 11
        human_mask = (seg_map == 11)
        label_image[human_mask] = 255
        return label_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model .tar path")
    parser.add_argument("--input_dir", type=str, help="Path to the directory of input images")
    parser.add_argument("--output_dir", type=str, help="Path where to put predictions")
    parser.add_argument("--resize", type=int,
                        help="You may specify the image size your model expects ('crop_size' param during export)")
    parser.add_argument("--no_color", action="store_true", help="Predict only classes, do not vonvert to colors")
    parser.add_argument("--colormap", type=str, choices=['milrem', 'cityscapes'], default='cityscapes')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    deeplab = DeepLab(args.model, colormap=args.colormap)

    if args.resize:
        deeplab.INPUT_SIZE = args.resize

    images = os.listdir(args.input_dir)
    for i, image_name in enumerate(images):
        image_path = os.path.join(args.input_dir, image_name)
        print(image_path)
        with open(image_path, 'rb') as fp:
            img = Image.open(fp).convert('RGB')
            width, height = img.size
        if args.no_color:
            pred = deeplab.predict_classes(img)
        else:  
            pred = deeplab.predict_image(img)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        # Upscale prediction to the original image
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(args.output_dir, image_name), pred)
