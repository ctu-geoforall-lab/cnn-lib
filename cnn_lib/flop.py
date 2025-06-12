#!/usr/bin/python3

import os
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping

# imports from this package
import cnn_lib.utils as utils

from cnn_lib.lib import AugmentGenerator
from cnn_lib.architectures import create_model
from cnn_lib.visualization import write_stats


def main(data_dir, model, in_weights_path=None, tensor_shape=(256, 256), backbone=None):

    # get nr of bands
    nr_bands = utils.get_nr_of_bands(data_dir)

    label_codes, label_names, id2code = utils.get_codings(
        os.path.join(data_dir, 'label_colors.txt'))

    print(model, backbone)
    model = create_model(
        model, len(id2code), nr_bands, tensor_shape, backbone=backbone, verbose=0
        )

    # val generator used for both the training and the detection
    val_generator = AugmentGenerator(
        data_dir, 1, 'val', tensor_shape, False,
        )

    model.load_weights(in_weights_path)
    testing_gen = val_generator(id2code, 0)
    batch_img, batch_mask = next(testing_gen)

    get_flops(model, batch_img)


def get_flops(model, model_inputs) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        """
        # if not hasattr(model, "model"):
        #     raise wandb.Error("self.model must be set before using this method.")

        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        # print(model_inputs)
        # print(tf.convert_to_tensor(model_inputs[0]).shape)
        # print(list(model_inputs[0].shape))
        inputs = [
                tf.TensorSpec([batch_size] + tf.convert_to_tensor(inp).shape, inp.dtype)
            for inp in model_inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(inputs[0])
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        print(flops.total_float_ops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run training or fine-tuning')

    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the directory containing images and labels')
    parser.add_argument(
        '--model', type=str, default='U-Net',
        choices=('U-Net', 'SegNet', 'DeepLab', 'FCN'),
        help='Model architecture')
    parser.add_argument(
        '--weights_path', type=str, default=None,
        help='ONLY FOR OPERATION == FINE-TUNE: Input weights path')
    parser.add_argument(
        '--tensor_height', type=int, default=256,
        help='Height of the tensor representing the image')
    parser.add_argument(
        '--tensor_width', type=int, default=256,
        help='Width of the tensor representing the image')
    parser.add_argument(
        '--backbone', type=str, default=None,
        choices=('ResNet50', 'ResNet101', 'ResNet152', 'VGG16'),
        help='Backbone architecture')

    args = parser.parse_args()

    main(args.data_dir,
         args.model,args.weights_path, (args.tensor_height, args.tensor_width), args.backbone)
