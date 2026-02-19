#!/usr/bin/python3

import os
import sys
import argparse

# imports from this package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if not '--help' in sys.argv and not '-h' in sys.argv:
    # in order to skip the slow import of tensorflow if not needed
    import cnn_lib.utils as utils
    bool_ = utils.str2bool
else:
    bool_ = bool


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
        '--input_regex', type=str, default='*.tif',
        help='Regex to be used to filter data supposed to be used for training.'
             'The images still have to have "image" in their names while labels'
             'have to have "label" in their names.')
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
        '--force_dataset_generation', type=bool_, default=False,
        help='Boolean to force the dataset structure generation')
    parser.add_argument(
        '--fit_dataset_in_memory', type=bool_, default=False,
        help='Boolean to load the entire dataset into memory instead '
             'of opening new files with each request - results in the '
             'reduction of I/O operations and time, but could result in huge '
             'memory needs in case of a big dataset')
    parser.add_argument(
        '--augment_training_dataset', type=bool_, default=False,
        help='Boolean to augment the training dataset with rotations, '
             'shear and flips')
    parser.add_argument(
        '--padding_mode', type=str, default=None,
        choices=('constant', 'reflect', 'symmetric'),
        help='Padding mode for edge tiles ("reflect", "symmetric", "constant"), '
             'or None for no padding (shift window behavior).')

    parser.add_argument(
        '--mask_ignore_value', type=int, default=255,
        help='Label value for padded mask regions (default 255)')

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

    from cnn_lib.train import run

    run(args.operation, args.data_dir, args.output_dir, args.model,
        args.model_fn, args.input_regex, args.weights_path,
        args.visualization_path, args.nr_epochs, args.initial_epoch,
        args.batch_size, args.loss_function, args.seed, args.patience,
        (args.tensor_height, args.tensor_width), args.monitored_value,
        args.force_dataset_generation, args.fit_dataset_in_memory,
        args.augment_training_dataset, args.padding_mode, args.mask_ignore_value,
        args.tversky_alpha, args.tversky_beta, args.dropout_rate_input,
        args.dropout_rate_hidden, args.validation_set_percentage,
        args.filter_by_classes, args.backbone)
