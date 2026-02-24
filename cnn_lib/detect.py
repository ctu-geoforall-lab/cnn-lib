#!/usr/bin/python3

import os
import glob

import tensorflow as tf

from osgeo import gdal

# imports from this package
import cnn_lib.utils as utils

from cnn_lib.lib import AugmentGenerator
from cnn_lib.architectures import create_model
from cnn_lib.visualization import visualize_detections
from cnn_lib.cnn_exceptions import DatasetError


def run(data_dir, model, in_weights_path, input_regex='*.tif',
        visualization_path='/tmp', batch_size=1, seed=1,
        tensor_shape=(256, 256), force_dataset_generation=False,
        fit_memory=False, val_set_pct=1, filter_by_class=None, backbone=None,
        ignore_masks=False, padding_mode=None, mask_ignore_value=255):
    utils.print_device_info()

    # get nr of bands
    val_images_dir = os.path.join(data_dir, 'val_images')
    if force_dataset_generation or not os.path.isdir(val_images_dir):
        nr_bands = utils.get_nr_of_bands(data_dir, input_regex)
    else:
        nr_bands = utils.get_nr_of_bands(val_images_dir, input_regex)

    if ignore_masks is False:
        val_masks_dir = os.path.join(data_dir, 'val_masks')
        if force_dataset_generation or not os.path.isdir(val_masks_dir):
            search_dir = data_dir
        else:
            search_dir = val_masks_dir

        filtered_files = glob.glob(os.path.join(search_dir, f'*{input_regex}*'))
        labels = [i for i in filtered_files if 'label' in i]
        if len(labels) == 0:
            raise DatasetError('No labels provided in the dataset.')

    label_codes, label_names, id2code = utils.get_codings(
        os.path.join(data_dir, 'label_colors.txt'))

    # set TensorFlow seed
    if seed is not None:
        import sys
        if int(tf.__version__.split('.')[1]) < 4:
            tf.random.set_seed(seed)
        else:
            tf.keras.utils.set_random_seed(seed)

    model = create_model(model, len(id2code), nr_bands, tensor_shape,
                         backbone=backbone, padding_mode=padding_mode,
                         mask_ignore_value=mask_ignore_value, verbose=1)

    # val generator used for both the training and the detection
    val_generator = AugmentGenerator(
        data_dir, input_regex, batch_size, 'val', tensor_shape, force_dataset_generation,
        fit_memory, val_set_pct=val_set_pct, filter_by_class=filter_by_class,
        ignore_masks=ignore_masks, padding_mode=padding_mode,
        mask_ignore_value=mask_ignore_value)

    # load weights
    model.load_weights(in_weights_path)
    model.set_weights(utils.model_replace_nans(model.get_weights()))

    detect(model, val_generator, id2code, [i for i in label_codes],
           label_names, data_dir, seed, visualization_path,
           ignore_masks=ignore_masks)


def detect(model, val_generator, id2code, label_codes, label_names,
           data_dir, seed=1, out_dir='/tmp', ignore_masks=False):
    """Run detection.

    :param model: model to be used for the detection
    :param val_generator: validation data generator
    :param id2code: dictionary mapping label ids to their codes
    :param label_codes: list with label codes
    :param label_names: list with label names
    :param data_dir: path to the directory containing images and labels
    :param seed: the generator seed
    :param out_dir: directory where the output visualizations will be saved
    :param ignore_masks: if computing average statistics (True) or running only
        prediction (False)
    """
    testing_gen = val_generator(id2code, seed)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get information needed to write referenced geotifs of detections
    geoinfos = get_geoinfo(val_generator.images_dir)

    batch_size = val_generator.batch_size

    for i in range(0, val_generator.nr_samples, batch_size):
        # get a batch of data
        batch_img, batch_mask = next(testing_gen)
        pred_all = model.predict(batch_img)

        batch_geoinfos = geoinfos[i:i + batch_size]

        # visualize the natch
        visualize_detections(batch_img, batch_mask, pred_all, id2code,
                             label_codes, label_names, batch_geoinfos,
                             out_dir, ignore_masks=ignore_masks)


def get_geoinfo(data_dir):
    """Get information needed to write referenced geotifs of detections.

    :param data_dir: path to the directory with either val_images
        or val_masks
    :return: list of sets in format [(filenames, projs, geo_transforms), ...]
    """
    filenames = []
    projs = []
    geo_transforms = []

    for filename in sorted(os.listdir(data_dir)):
        src = gdal.Open(os.path.join(data_dir, filename), gdal.GA_ReadOnly)
        filenames.append(filename)
        projs.append(src.GetProjection())
        geo_transforms.append(src.GetGeoTransform())

        src = None

    return [i for i in zip(filenames, projs, geo_transforms)]
