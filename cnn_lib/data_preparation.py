#!/usr/bin/python3

import os
import glob
import shutil

import numpy as np

from osgeo import gdal

from cnn_lib.cnn_exceptions import DatasetError


def generate_dataset_structure(data_dir, input_regex, tensor_shape=(256, 256),
                               val_set_pct=0.2, filter_by_class=None,
                               augment=True, ignore_masks=False,padding_mode=None, mask_ignore_value=255, verbose=1):
    """Generate the expected dataset structure.

    Will generate directories train_images, train_masks, val_images and
    val_masks.

    :param data_dir: path to the directory containing images
    :param input_regex: regex to be used to filter data supposed to be used
        for training
    :param tensor_shape: shape of the first two dimensions of input tensors
    :param val_set_pct: percentage of the validation images in the dataset
    :param filter_by_class: classes of interest (if specified, only samples
        containing at least one of them will be created)
    :param augment: boolean saying whether to augment the dataset or not
    :param ignore_masks: do not create masks
    :param padding_mode: padding mode for edge tiles ('reflect', 'symmetric',
        'edge', 'constant', or None for no padding - shift window behavior)
    :param mask_ignore_value: label value for padded mask regions (default 255)
    :param verbose: verbosity (0=quiet, >0 verbose)
    """
    # Create folders to hold images and masks
    if ignore_masks is False:
        dirs = ('train_images', 'train_masks', 'val_images', 'val_masks')
    else:
        dirs = ('train_images', 'val_images')

    for directory in dirs:
        dir_full_path = os.path.join(data_dir, directory)
        if os.path.isdir(dir_full_path):
            shutil.rmtree(dir_full_path)

        os.makedirs(dir_full_path)

    dir_names = train_val_determination(val_set_pct)

    # tile and write samples
    filtered_files = sorted(
        glob.glob(os.path.join(data_dir, f'*{input_regex}*')))
    source_images = [i for i in filtered_files if 'image' in i]
    for i in source_images:
        tile(i, i.replace('image', 'label'), tensor_shape,
             filter_by_class, augment, dir_names, ignore_masks, padding_mode, mask_ignore_value)

    # check if there are some training data
    train_images_nr = len(os.listdir(os.path.join(data_dir, 'train_images')))
    val_images_nr = len(os.listdir(os.path.join(data_dir, 'val_images')))
    if train_images_nr + val_images_nr == 0:
        raise DatasetError('No training samples created. Check the size of '
                           'the images in the data_dir or the appearance of '
                           'the classes you are interested in in labels')
    elif verbose > 0:
        print('Created {} training and {} validation samples from {} '
              'provided image(s).'.format(train_images_nr, val_images_nr,
                                          len(source_images)))


def tile(scene_path, labels_path, tensor_shape, filter_by_class=None,
         augment=True, dir_names=None, ignore_masks=False, padding_mode=None, mask_ignore_value=255 ):
    """Tile the big scene into smaller samples and write them.

    If filter_by_class is not None, only samples containing at least one of
    these classes of interest will be returned.
    If augment is True, data are augmented by every sample being rotated by
    90, 180, and 270 degrees.

    :param scene_path: path to the image to be cut
    :param labels_path: path to the image with labels to be cut
    :param tensor_shape: shape of the first two dimensions of input tensors
    :param filter_by_class: classes of interest (if specified, only samples
        containing at least one of them will be returned)
    :param augment: boolean saying whether to augment the dataset or not
    :param dir_names: a generator determining directory names (train/val)
    :param ignore_masks: do not create masks
    :param padding_mode: padding mode for edge tiles ('reflect', 'symmetric',
        'edge', 'constant', or None for no padding - shift window behavior)
    :param mask_ignore_value: label value for padded mask regions (default 255)
    """
    rows_step = tensor_shape[0]
    cols_step = tensor_shape[1]

    # do we filter by classes?
    if filter_by_class is None:
        filt = False
    else:
        filter_by_class = [int(i) for i in filter_by_class.split(',')]
        filt = True

    # the following variables are defined here to avoid creating them in the
    # loop later
    driver = gdal.GetDriverByName("GTiff")
    scene = gdal.Open(scene_path, gdal.GA_ReadOnly)
    nr_bands = scene.RasterCount
    projection = scene.GetProjection()
    data_type = scene.GetRasterBand(1).DataType
    nr_rows = scene.RasterYSize
    nr_cols = scene.RasterXSize
    scene = None

    if cols_step == rows_step:
        rotations = (1, 2, 3)
    else:
        rotations = (2, )

    # do not write aux.xml files
    os.environ['GDAL_PAM_ENABLED'] = 'NO'

    # get variables for the loop checks
    if ignore_masks is False:
        labels = gdal.Open(labels_path, gdal.GA_ReadOnly)
        labels_np = labels.GetRasterBand(1).ReadAsArray()
        labels = None
    else:
        labels_np = None

    scene_dir, scene_name = os.path.split(scene_path[:-10])

    for i in range(0, nr_cols, cols_step):
        if padding_mode is None:
            # shift window
            # if reaching the end of the image, expand the window back to
            # avoid pixels outside the image
            if i + cols_step > nr_cols:
                i = nr_cols - cols_step
            actual_cols = cols_step
            right_pad = 0

        else:
            # crop what is available and add padding if needed
            if i + cols_step > nr_cols:
                actual_cols = nr_cols - i
                right_pad = cols_step - actual_cols
            else:
                actual_cols = cols_step
                right_pad = 0

        for j in range(0, nr_rows, rows_step):
            if padding_mode is None:
                # shift window
                # if reaching the end of the image, expand the window back to
                # avoid pixels outside the image
                if j + rows_step > nr_rows:
                    j = nr_rows - rows_step
                actual_rows = rows_step
                bottom_pad = 0
            else:
                # crop what is available and add padding if needed
                if j + rows_step > nr_rows:
                    actual_rows = nr_rows - j
                    bottom_pad = rows_step - actual_rows
                else:
                    actual_rows = rows_step
                    bottom_pad = 0

            # if filtering, check if it makes sense to continue
            if filt is True and ignore_masks is False:
                labels_cropped = labels_np[j:j + actual_rows, i:i + actual_cols]
                if not any(i in labels_cropped for i in filter_by_class):
                    # no occurrence of classes to filter by - continue with
                    # next patch
                    continue

            # CROPPING SECTION

            dir_name = next(dir_names)

            # get paths
            output_scene_path = os.path.join(scene_dir,
                                             '{}_images'.format(dir_name),
                                             scene_name + f'_{i}_{j}.tif')
            # check if padding is needed
            pad_needed = right_pad>0 or bottom_pad>0

            scene_src = gdal.Open(scene_path, gdal.GA_ReadOnly)
            raw_geo = scene_src.GetGeoTransform()
            # read all bands at once as a (bands, rows, cols) array
            scene_array = scene_src.ReadAsArray(i, j, actual_cols, actual_rows)
            if pad_needed:
                scene_array = np.pad(scene_array, ((0, 0), (0, bottom_pad), (0, right_pad)), mode=padding_mode)
            scene_src = None
            scene_bands = list(scene_array)

            geo_transform = list(raw_geo)
            geo_transform[0] = raw_geo[0] + i * raw_geo[1]
            geo_transform[3] = raw_geo[3] + j * raw_geo[5]

            if ignore_masks is False:
                mask_src = gdal.Open(labels_path, gdal.GA_ReadOnly)
                mask_array = mask_src.GetRasterBand(1).ReadAsArray(
                    i, j, actual_cols, actual_rows)
                mask_src = None
                if pad_needed:
                    mask_array = np.pad(
                        mask_array,
                        ((0, bottom_pad), (0, right_pad)),
                        mode='constant', constant_values=mask_ignore_value)
            else:
                mask_array = None

            # unified loop over original + rotations
            all_rotations = [0] + list(rotations) if augment else [0]
            for rot_k in all_rotations:
                dir_name = next(dir_names)
                suffix = f'_rot{rot_k * 90}' if rot_k > 0 else ''

                out_scene_path = os.path.join(
                    scene_dir, '{}_images'.format(dir_name),
                    scene_name + f'_{i}_{j}{suffix}.tif')
                out_scene = driver.Create(out_scene_path,
                                          cols_step, rows_step,
                                          nr_bands, data_type)
                out_scene.SetGeoTransform(geo_transform)
                out_scene.SetProjection(projection)
                for band_i in range(nr_bands):
                    band_array = scene_array[band_i]
                    if rot_k > 0:
                        band_array = np.rot90(band_array, rot_k)
                    out_scene.GetRasterBand(band_i + 1).WriteArray(band_array)
                out_scene = None

                if ignore_masks is False:
                    out_mask_path = os.path.join(
                        scene_dir, '{}_masks'.format(dir_name),
                        scene_name + f'_{i}_{j}{suffix}.tif')
                    out_mask = driver.Create(out_mask_path,
                                             cols_step, rows_step,
                                             1, gdal.GDT_UInt16)
                    out_mask.SetGeoTransform(geo_transform)
                    out_mask.SetProjection(projection)
                    mask_to_write = mask_array
                    if rot_k > 0:
                        mask_to_write = np.rot90(mask_array, rot_k)
                    out_mask.GetRasterBand(1).WriteArray(mask_to_write)
                    out_mask = None


def train_val_determination(pct):
    """Return the decision if the sample will be part of the train or val set.

    :param pct: Percentage at which a val determinator is returned
    """
    cur_pct = 0
    while True:
        cur_pct += pct
        if cur_pct < 1:
            yield 'train'
        else:
            cur_pct -= 1
            yield 'val'
