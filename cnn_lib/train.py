#!/usr/bin/python3

import os
import math

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping

# imports from this package
import cnn_lib.utils as utils

from cnn_lib.lib import AugmentGenerator
from cnn_lib.architectures import create_model
from cnn_lib.visualization import write_stats


def run(operation, data_dir, output_dir, model, model_fn, in_weights_path=None,
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

    model = create_model(
        model, len(id2code), nr_bands, tensor_shape, loss=loss_function,
        alpha=tversky_alpha, beta=tversky_beta,
        dropout_rate_input=dropout_rate_input,
        dropout_rate_hidden=dropout_rate_hidden, backbone=backbone, name=name)

    # val generator used for both the training and the detection
    val_generator = AugmentGenerator(
        data_dir, batch_size, 'val', tensor_shape, force_dataset_generation,
        fit_memory, augment=augment, val_set_pct=val_set_pct,
        filter_by_class=filter_by_class, verbose=verbose)

    # load weights if the model is supposed to do so
    if operation == 'fine-tune':
        model.load_weights(in_weights_path)

    train_generator = AugmentGenerator(
        data_dir, batch_size, 'train', fit_memory=fit_memory,
        augment=augment)
    train(model, train_generator, val_generator, id2code, batch_size,
          output_dir, visualization_path, model_fn, nr_epochs,
          initial_epoch, seed=seed, patience=patience,
          monitored_value=monitored_value, verbose=verbose)


def train(model, train_generator, val_generator, id2code, batch_size,
          output_dir, visualization_path, model_fn, nr_epochs,
          initial_epoch=0, seed=1, patience=100,
          monitored_value='val_accuracy', verbose=1):
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
    # set up model_path
    if model_fn is None:
        model_fn = '{}_ep{}_pat{}.h5'.format(model.lower(), nr_epochs,
                                             patience)

    out_model_path = os.path.join(output_dir, model_fn)

    # set up log dir
    log_dir = os.path.join(output_dir, 'logs')

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
    steps_per_epoch = math.ceil(train_generator.nr_samples / batch_size)
    validation_steps = math.ceil(val_generator.nr_samples / batch_size)

    # train
    result = model.fit(
        train_generator(id2code, seed),
        validation_data=val_generator(id2code, seed),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=nr_epochs,
        initial_epoch=initial_epoch,
        verbose=verbose,
        callbacks=callbacks)

    write_stats(result, os.path.join(visualization_path, 'accu.png'))
