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
            if not pad_needed:
                # create tile directly
                gdal.Translate(output_scene_path,
                               scene_path,
                               srcWin=(i, j, cols_step, rows_step))

                if ignore_masks is False:
                    # do the same for masks
                    output_mask_path = os.path.join(scene_dir,
                                                    '{}_masks'.format(dir_name),
                                                    scene_name + f'_{i}_{j}.tif')
                    gdal.Translate(output_mask_path,
                                   labels_path,
                                   srcWin=(i, j, cols_step, rows_step))
            else:
                # crop, add padding and then save
                scene_src = gdal.Open(scene_path, gdal.GA_ReadOnly)
                scene_bands = []
                for band_i in range(1, nr_bands + 1):
                    # read band
                    band_array = scene_src.GetRasterBand(band_i).ReadAsArray(
                        i, j, actual_cols, actual_rows)
                    # apply padding
                    padded_band = np.pad(band_array, ((0, bottom_pad), (0, right_pad)), mode=padding_mode)
                    scene_bands.append(padded_band)

                geo_transform = scene_src.GetGeoTransform()
                # Adjust geotransform for the cropped origin
                new_geo_transform = list(geo_transform)
                new_geo_transform[0] = geo_transform[0] + i * geo_transform[1]
                new_geo_transform[3] = geo_transform[3] + j * geo_transform[5]
                scene_src = None

                # Write padded image
                out_scene = driver.Create(output_scene_path, cols_step, rows_step, nr_bands, data_type)
                out_scene.SetGeoTransform(new_geo_transform)
                out_scene.SetProjection(projection)

                for band_i in range(nr_bands):
                    out_band = out_scene.GetRasterBand(
                        band_i + 1)
                    out_band.WriteArray(scene_bands[band_i], 0, 0)

                out_scene = None

                if ignore_masks is False:
                    mask_src = gdal.Open(labels_path, gdal.GA_ReadOnly)
                    mask_array = mask_src.GetRasterBand(1).ReadAsArray(
                        i, j, actual_cols, actual_rows)
                    mask_src = None

                    padded_mask = np.pad(mask_array,((0, bottom_pad), (0, right_pad)),mode='constant',constant_values=mask_ignore_value)

                    output_mask_path = os.path.join(
                        scene_dir,
                        '{}_masks'.format(dir_name),
                        scene_name + f'_{i}_{j}.tif'
                    )
                    out_mask = driver.Create(output_mask_path, cols_step, rows_step, 1, gdal.GDT_UInt16)
                    out_mask.SetGeoTransform(new_geo_transform)
                    out_mask.SetProjection(projection)
                    out_mask_band = out_mask.GetRasterBand(1)
                    out_mask_band.WriteArray(padded_mask, 0, 0)
                    out_mask = None

            if augment is False:
                # the following code is unnecessary then
                continue

            # AUGMENTATION SECTION

            # get info (in the loop because we want the geotransform
            # of the cropped image)
            src_scene = gdal.Open(output_scene_path, gdal.GA_ReadOnly)
            geo_transform = src_scene.GetGeoTransform()
            src_bands = []
            for band_i in range(1, nr_bands + 1):
                src_bands.append(
                    src_scene.GetRasterBand(band_i).ReadAsArray())

            if ignore_masks is False:
                src_mask = gdal.Open(output_mask_path, gdal.GA_ReadOnly)
                src_mask_band = src_mask.GetRasterBand(1).ReadAsArray()
            else:
                src_mask_band = None

            src_scene = None
            src_mask = None

            for rot_k in rotations:
                dir_name = next(dir_names)

                # add 'rot_{X}deg' to the filename
                rot_scene_path = os.path.join(
                    scene_dir, '{}_images'.format(dir_name),
                    scene_name + f'_{i}_{j}_rot{rot_k * 90}.tif')

                # create files
                out_scene = driver.Create(
                    rot_scene_path,
                    cols_step,
                    rows_step,
                    nr_bands,
                    data_type)

                out_scene.SetGeoTransform(geo_transform)
                out_scene.SetProjection(projection)

                # write rotated arrays
                for band_i in range(nr_bands):
                    out_scene_band = out_scene.GetRasterBand(
                        band_i + 1)
                    out_scene_band.WriteArray(
                        np.rot90(src_bands[band_i], rot_k), 0, 0)

                if ignore_masks is False:
                    # do the same for masks
                    rot_mask_path = os.path.join(
                        scene_dir, '{}_masks'.format(dir_name),
                        scene_name + f'_{i}_{j}_rot{rot_k * 90}.tif')
                    out_mask = driver.Create(
                        rot_mask_path,
                        cols_step,
                        rows_step,
                        1,
                        gdal.GDT_UInt16)
                    out_mask.SetGeoTransform(geo_transform)
                    out_mask.SetProjection(projection)
                    out_mask_band = out_mask.GetRasterBand(1)
                    out_mask_band.WriteArray(
                        np.rot90(src_mask_band, rot_k), 0, 0)

                out_scene = None
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
