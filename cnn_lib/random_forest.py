#!/usr/bin/python3

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from osgeo import gdal_array

import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from osgeo import gdal
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping
from tensorflow.math import confusion_matrix
from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy

# imports from this package
import utils

from cnn_lib import AugmentGenerator
from architectures import create_model
from visualization import write_stats
from cnn_lib import categorical_dice


def main(operation, data_dir, output_dir, model, model_fn, in_weights_path=None,
         visualization_path='/tmp', nr_epochs=1, initial_epoch=0, batch_size=1,
         loss_function='dice', seed=1, patience=100, tensor_shape=(256, 256),
         monitored_value='val_accuracy', force_dataset_generation=False,
         fit_memory=False, augment=False, tversky_alpha=0.5,
         tversky_beta=0.5, dropout_rate_input=None, dropout_rate_hidden=None,
         val_set_pct=0.2, filter_by_class=None, backbone=None, name='model',
         verbose=1):
    if verbose > 0:
        utils.print_device_info()

    # get nr of bands
    nr_bands = utils.get_nr_of_bands(data_dir)

    label_codes, label_names, id2code = utils.get_codings(
        os.path.join(data_dir, 'label_colors.txt'))

    # set TensorFlow seed
    if seed is not None:
        import sys
        if int(tf.__version__.split('.')[1]) < 4:
            tf.random.set_seed(seed)
        else:
            tf.keras.utils.set_random_seed(seed)

    model = tfdf.keras.RandomForestModel(check_dataset=False)

    # val generator used for both the training and the detection
    val_generator = AugmentGenerator(
        data_dir, batch_size, 'val', tensor_shape, force_dataset_generation,
        fit_memory, augment=augment, val_set_pct=val_set_pct,
        filter_by_class=filter_by_class, verbose=verbose)
    train_generator = AugmentGenerator(
        data_dir, batch_size, 'train', tensor_shape, force_dataset_generation,
        fit_memory, augment=augment, val_set_pct=val_set_pct, onehot_encode=False,
        filter_by_class=filter_by_class, verbose=verbose)

    # load weights if the model is supposed to do so
    if operation == 'fine-tune':
        model.load_weights(in_weights_path)

    print('**' * 80)
    train_images = []
    train_masks = []
    train_generator = []
    val_generator = []
    for i in glob.glob(os.path.join(data_dir, 'train_images', '*')):
        train_generator.append([])
        orig_image = gdal_array.LoadFile(i).astype('int16')
        train_image = gdal_array.LoadFile(i).astype('int16').flatten()
        # train_image = np.moveaxis(gdal_array.LoadFile(i), 0, -1).astype('int16').flatten()
        train_generator[-1].append(np.moveaxis(gdal_array.LoadFile(i), 0, -1).astype('int16'))
    # train_image = np.stack(train_images)
    a = 0
    for i in glob.glob(os.path.join(data_dir, 'train_masks', '*')):
        train_mask = np.repeat(gdal_array.LoadFile(i).astype('int16').flatten(), orig_image.shape[0])
        # train_mask = np.moveaxis(gdal_array.LoadFile(i), 0, -1).astype('int16').flatten()
        train_generator[a].append(gdal_array.LoadFile(i).astype('uint8'))
        a += 1
    # train_mask = np.stack(train_masks)
    # train_generator = tf.data.Dataset.from_tensor_slices((train_image, train_mask))
    for i in glob.glob(os.path.join(data_dir, 'val_images', '*')):
        val_generator.append([])
        val_generator[-1].append(np.moveaxis(gdal_array.LoadFile(i), 0, -1).astype('int16'))
    a = 0
    for i in glob.glob(os.path.join(data_dir, 'val_masks', '*')):
        val_generator[a].append(gdal_array.LoadFile(i).astype('uint8'))
        a += 1
    # val_generator = tf.data.Dataset.from_tensor_slices((val_image, val_mask))
    # for i in val_generator:
    #     print('***')
    #     print(i)

    train(model, train_generator, val_generator, id2code, batch_size,
          output_dir, visualization_path, model_fn, nr_epochs,
          initial_epoch, seed=seed, patience=patience,
          monitored_value=monitored_value, verbose=verbose, label_codes=label_codes, label_names=label_names, data_dir=data_dir)


def train(model, train_generator, val_generator, id2code, batch_size,
          output_dir, visualization_path, model_fn, nr_epochs,
          initial_epoch=0, seed=1, patience=100,
          monitored_value='val_accuracy', verbose=1, label_codes='', label_names='', data_dir=''):
    """Run model training.

    :param model: model to be used for the detection
    :param train_generator: training data generator
    :param val_generator: validation data generator
    :param id2code: dictionary mapping label ids to their codes
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :param output_dir: path where logs and the model will be saved
    :param visualization_path: path to a directory where the output
        visualizations will be saved
    :param model_fn: model file name
    :param nr_epochs: number of epochs to train the model
    :param initial_epoch: epoch at which to start training
    :param seed: the generator seed
    :param patience: number of epochs with no improvement after which training
        will be stopped
    :param monitored_value: metric name to be monitored
    :param verbose: verbosity (0=quiet, >0 verbose)
    """
    model_fn = 'randomforest.h5'

    out_model_path = os.path.join(output_dir, model_fn)

    # set up log dir
    log_dir = os.path.join(output_dir, 'logs')
    import pip
    pip.main(['install', 'tensorflow-addons'])
    import tensorflow_addons as tfa

    # create output_dir if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get the correct early stop mode
    if 'accuracy' in monitored_value:
        early_stop_mode = 'max'
    else:
        early_stop_mode = 'min'

    # set up monitoring
    tb = TensorBoard(log_dir=log_dir, write_graph=True)
    mc = ModelCheckpoint(
        mode=early_stop_mode, filepath=out_model_path,
        monitor=monitored_value, save_best_only='True',
        save_weights_only='True',
        verbose=1)
    # TODO: check custom earlystopping to monitor multiple metrics
    #       https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
    es = EarlyStopping(mode=early_stop_mode, monitor=monitored_value,
                       patience=patience, verbose=verbose, restore_best_weights=True)
    callbacks = [tb, mc, es]

    # steps per epoch not needed to be specified if the data are augmented, but
    # not when they are not (our own generator is used)
    # steps_per_epoch = np.ceil(train_generator.nr_samples / batch_size)
    # validation_steps = np.ceil(val_generator.nr_samples / batch_size)

    train
    model.save_weights(out_model_path[:-3] + '_pred.h5')
    for image, mask in train_generator:
        train_generato = tf.data.Dataset.from_tensor_slices((image, mask))
        result = model.fit(
            # train_generator(id2code, seed),)
            train_generato,
            # validation_data=val_generator(id2code, seed),
            # validation_data=val_generator,
            # # steps_per_epoch=steps_per_epoch,
            # # validation_steps=validation_steps,
            # epochs=nr_epochs,
            # initial_epoch=initial_epoch,
            # verbose=verbose,
            callbacks=callbacks)

    import glob
    from osgeo import gdal_array
    # val_image = np.moveaxis(gdal_array.LoadFile(os.path.join(data_dir, 'val_images/361_5621_7616_7072_rot90.tif')), 0, -1).astype('int16')
    # val_mask = gdal_array.LoadFile(os.path.join(data_dir, 'val_masks/361_5621_7616_7072_rot90.tif')).astype('int8')
    val_image = np.moveaxis(gdal_array.LoadFile(os.path.join(data_dir, 'train_images/s01_20190423_0_0_1024_1024.tif')), 0, -1).astype('int16')
    val_mask = gdal_array.LoadFile(os.path.join(data_dir, 'train_masks/s01_20190423_0_0_1024_1024.tif')).astype('int8')
    val_generato = tf.data.Dataset.from_tensor_slices((val_image, val_mask))
    batch_geoinfos = get_geoinfo(os.path.join(data_dir, 'val_images'))
    pred_all = model.predict(val_generato)
    import tensorflow_addons as tfa
    f1 = tfa.metrics.F1Score(val_mask.shape[1], average='macro')
    dim = 1024  # 544  # 352
    if pred_all.shape[1] == 1: 
        detection_decoded = (pred_all > 0.8).reshape(dim, dim)
    else:
        detection_decoded = np.argmax(pred_all.reshape(dim, dim, pred_all.shape[1]), axis=-1).reshape(dim, dim)
    f1.update_state(val_mask.astype('float'), detection_decoded.astype('float'))
    print('***************')
    print('id2code')
    print(id2code)
    print('pred_all.shape')
    print(pred_all.shape)
    print('f1')
    print(f1.result())

    visualize_detections([val_image], [val_mask], [pred_all], id2code,
                         label_codes, label_names, batch_geoinfos,
                         output_dir, ignore_masks=False, dim=dim)

def visualize_detections(images, ground_truths, detections, id2code,
                         label_codes, label_names, geoinfos, out_dir='/tmp',
                         ignore_masks=False, dim=352):
    """Create visualizations.

    Consist of the original image, the confusion matrix, ground truth labels
    and the model predicted labels.

    :param images: original images
    :param ground_truths: ground truth labels
    :param detections: the model label predictions
    :param id2code: dictionary mapping label ids to their codes
    :param label_codes: list with label codes
    :param label_names: list with label names
    :param geoinfos: list in format(filename, projection, geo_transform)
    :param out_dir: directory where the output visualizations will be saved
    :param ignore_masks: if computing average statistics (True) or running only
        prediction (False)
    """
    def dice_coef(y_true, y_pred):
        y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)
    max_id = max(id2code.values())
    name_range = range(len(label_names))

    driver = gdal.GetDriverByName("GTiff")
    plt.rcParams['figure.dpi'] = 300

    for i in range(0, 1):#np.shape(detections)[0]):
        if i == len(geoinfos):
            # the sample count is not dividable by batch_size
            break

        # THE OVERVIEW IMAGE SECTION

        fig = plt.figure(figsize=(17, 17))

        # original image
        ax1 = fig.add_subplot(2, 2, 1)
        # TODO: expect also other data than S2
        a = np.stack((images[i][:, :, 2], images[i][:, :, 1],
                      images[i][:, :, 0]), axis=2)
        ax1.imshow((255 / a.max() * a).astype(np.uint8))
        ax1.title.set_text('Actual image')

        # detections
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('Predicted labels')
        print('dice')
        if detections[i].shape[1] == 1: 
            detection_decoded = (detections[i] > 0.8).reshape(dim, dim, 1)
            print(categorical_dice(onehot_encode(ground_truths[i], id2code), onehot_encode(detection_decoded, id2code)))
            print(dice_coef(ground_truths[i].reshape(1, dim, dim), detection_decoded.reshape(1, dim, dim)))
        else:
            detection_decoded = np.argmax(detections[i].reshape(dim, dim, detections[i].shape[1]), axis=-1).reshape(dim, dim, 1)
            print(categorical_dice(onehot_encode(ground_truths[i], id2code), onehot_encode(detection_decoded, id2code)))
        pred_labels = detection_decoded
        ax4.imshow(pred_labels * 4)

        if ignore_masks is False:
            # ground truths
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_title('Ground truth labels')
            gt_labels = ground_truths[i]
            # gt_labels = onehot_decode(gt_labels, id2code)
            ax3.imshow(gt_labels * 4)

            # confusion matrix
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.set_title('Confusion matrix')
            conf_matrix = confusion_matrix(
                gt_labels[:, :].flatten(), pred_labels[:, :, 0].flatten(),
                max_id + 1)
            # subset to existing classes
            conf_matrix = conf_matrix.numpy()[label_codes][:, label_codes]
            # normalize the confusion matrix
            row_sums = conf_matrix.sum(axis=1)[:, np.newaxis]
            # TODO: solve division by 0
            cm_norm = np.around(
                conf_matrix.astype('float') / row_sums, decimals=2
            )
            # visualize
            ax2.imshow(cm_norm, cmap=plt.cm.Blues)
            y_labels = ['{}\n{}'.format(label_names[j], row_sums[j]) for j in
                        name_range]
            plt.xticks(name_range, label_names)
            plt.yticks(name_range, y_labels)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            # write percentage values (0.00 -- 1.00) into the confusion matrix
            threshold = cm_norm.max() / 2.  # used to decide for the font colour
            for row in range(len(conf_matrix)):
                for col in range(len(conf_matrix)):
                    if cm_norm[col, row] > threshold:
                        colour = 'white'
                    else:
                        colour = 'black'
                    # TODO: class names, not codes
                    ax2.text(row, col, cm_norm[col, row], color=colour,
                             horizontalalignment='center')

        # save the overview image
        plt.savefig(os.path.join(out_dir, geoinfos[i][0][:-4]), bbox_inches='tight')
        plt.close()

        # THE DETECTION TIF IMAGE SECTION

        print('overall accuracy')
        print(np.sum(gt_labels == pred_labels[:, :, 0]) / dim / dim)
        out = driver.Create(os.path.join(out_dir, f'{geoinfos[i][0]}'),
                            np.shape(detection_decoded)[0],
                            np.shape(detection_decoded)[1],
                            1,
                            gdal.GDT_Byte)
        outband = out.GetRasterBand(1)
        outband.WriteArray(detection_decoded[:, :, 0], 0, 0)
        out.SetProjection(geoinfos[i][1])
        out.SetGeoTransform(geoinfos[i][2])

        out = None


def onehot_encode(orig_image, colormap):
    """Encode input images into one hot ones.

    Unfortunately, keras.utils.to_categorical cannot be used because our
    classes are not consecutive.

    :param orig_image: original image
    :param colormap: dictionary mapping label ids to their codes
    :return: One hot encoded image of dimensions
        (height x width x num_classes)
    """
    num_classes = len(colormap)
    shape = orig_image.shape[:2] + (num_classes,)
    encoded_image = np.empty(shape, dtype=np.uint8)

    # reshape to the shape used inside the onehot matrix
    reshaped = orig_image.reshape((-1, 1))

    for i, cls in enumerate(colormap):
        all_ax = np.all(reshaped == colormap[i], axis=1)
        encoded_image[:, :, i] = all_ax.reshape(shape[:2])

    return encoded_image

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

def onehot_decode(onehot, colormap, nr_bands=3, enhance_colours=True):
    """Decode onehot mask labels to an eye-readable image.

    :param onehot: one hot encoded image matrix (height x width x
        num_classes)
    :param colormap: dictionary mapping label ids to their codes
    :param nr_bands: number of bands of intended input images
    :param enhance_colours: Enhance the contrast between colours
        (pseudorandom multiplication of the colour value)
    :return: decoded RGB image (height x width x 3)
    """
    # create 2D matrix with label ids (so you do not have to loop)
    single_layer = np.argmax(onehot, axis=-1)
    # import tensorflow as tf
    # single_layer = tf.argmax(onehot, axis=-1)

    # create colourful visualizations
    out_shape = (onehot.shape[0], onehot.shape[1], nr_bands)
    output = np.zeros(out_shape)
    # output = tf.zeros(out_shape)
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]

    if enhance_colours is True:
        multiply_vector = [i ** 3 for i in range(1, nr_bands + 1)]
        enhancement_matrix = np.ones(out_shape) * np.array(multiply_vector,
                                                           dtype=np.uint8)
        output *= enhancement_matrix

    return np.uint8(output)
    # return single_layer
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run training or fine-tuning')

    parser.add_argument(
        '--operation', type=str, default='train',
        choices=('train', 'fine-tune'),
        help='Choose either to train the model or to use a trained one for '
             'detection')
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the directory containing images and labels')
    parser.add_argument(
        '--output_dir', type=str, required=True, default=None,
        help='Path where logs and the model will be saved')
    parser.add_argument(
        '--model', type=str, default='U-Net',
        choices=('U-Net', 'SegNet', 'DeepLab', 'FCN'),
        help='Model architecture')
    parser.add_argument(
        '--model_fn', type=str,
        help='Output model filename')
    parser.add_argument(
        '--weights_path', type=str, default=None,
        help='ONLY FOR OPERATION == FINE-TUNE: Input weights path')
    parser.add_argument(
        '--visualization_path', type=str, default='/tmp',
        help='Path to a directory where the accuracy visualization '
             'will be saved')
    parser.add_argument(
        '--nr_epochs', type=int, default=1,
        help='Number of epochs to train the model. Note that in conjunction '
             'with initial_epoch, epochs is to be understood as the final '
             'epoch')
    parser.add_argument(
        '--initial_epoch', type=int, default=0,
        help='ONLY FOR OPERATION == FINE-TUNE: Epoch at which to start '
             'training (useful for resuming a previous training run)')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The number of samples that will be propagated through the '
             'network at once')
    parser.add_argument(
        '--loss_function', type=str, default='dice',
        choices=('binary_crossentropy', 'categorical_crossentropy', 'dice',
                 'tversky'),
        help='A function that maps the training onto a real number '
             'representing cost associated with the epoch')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Generator random seed')
    parser.add_argument(
        '--patience', type=int, default=100,
        help='Number of epochs with no improvement after which training will '
             'be stopped')
    parser.add_argument(
        '--tensor_height', type=int, default=256,
        help='Height of the tensor representing the image')
    parser.add_argument(
        '--tensor_width', type=int, default=256,
        help='Width of the tensor representing the image')
    parser.add_argument(
        '--monitored_value', type=str, default='val_accuracy',
        help='Metric name to be monitored')
    parser.add_argument(
        '--force_dataset_generation', type=utils.str2bool, default=False,
        help='Boolean to force the dataset structure generation')
    parser.add_argument(
        '--fit_dataset_in_memory', type=utils.str2bool, default=False,
        help='Boolean to load the entire dataset into memory instead '
             'of opening new files with each request - results in the '
             'reduction of I/O operations and time, but could result in huge '
             'memory needs in case of a big dataset')
    parser.add_argument(
        '--augment_training_dataset', type=utils.str2bool, default=False,
        help='Boolean to augment the training dataset with rotations, '
             'shear and flips')
    parser.add_argument(
        '--tversky_alpha', type=float, default=None,
        help='ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient alpha')
    parser.add_argument(
        '--tversky_beta', type=float, default=None,
        help='ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient beta')
    parser.add_argument(
        '--dropout_rate_input', type=float, default=None,
        help='Fraction of the input units of the  input layer to drop')
    parser.add_argument(
        '--dropout_rate_hidden', type=float, default=None,
        help='Fraction of the input units of the hidden layers to drop')
    parser.add_argument(
        '--validation_set_percentage', type=float, default=0.2,
        help='If generating the dataset - Percentage of the entire dataset to '
             'be used for the validation or detection in the form of '
             'a decimal number')
    parser.add_argument(
        '--filter_by_classes', type=str, default=None,
        help='If generating the dataset - Classes of interest. If specified, '
             'only samples containing at least one of them will be created. '
             'If filtering by multiple classes, specify their values '
             'comma-separated (e.g. "1,2,6" to filter by classes 1, 2 and 6)')
    parser.add_argument(
        '--backbone', type=str, default=None,
        choices=('ResNet50', 'ResNet101', 'ResNet152', 'VGG16'),
        help='Backbone architecture')

    args = parser.parse_args()

    # check required arguments by individual operations
    if args.operation == 'fine-tune' and args.weights_path is None:
        raise parser.error(
            'Argument weights_path required for operation == fine-tune')
    if args.operation == 'train' and args.initial_epoch != 0:
        raise parser.error(
            'Argument initial_epoch must be 0 for operation == train')
    tversky_none = None in (args.tversky_alpha, args.tversky_beta)
    if args.loss_function == 'tversky' and tversky_none is True:
        raise parser.error(
            'Arguments tversky_alpha and tversky_beta must be set for '
            'loss_function == tversky')
    dropout_specified = args.dropout_rate_input is not None or \
                        args.dropout_rate_hidden is not None
    if not 0 <= args.validation_set_percentage < 1:
        raise parser.error(
            'Argument validation_set_percentage must be greater or equal to '
            '0 and smaller or equal than 1')

    main(args.operation, args.data_dir, args.output_dir,
         args.model, args.model_fn, args.weights_path, args.visualization_path,
         args.nr_epochs, args.initial_epoch, args.batch_size,
         args.loss_function, args.seed, args.patience,
         (args.tensor_height, args.tensor_width), args.monitored_value,
         args.force_dataset_generation, args.fit_dataset_in_memory,
         args.augment_training_dataset, args.tversky_alpha,
         args.tversky_beta, args.dropout_rate_input,
         args.dropout_rate_hidden, args.validation_set_percentage,
         args.filter_by_classes, args.backbone)
