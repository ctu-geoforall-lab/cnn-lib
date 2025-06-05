#!/usr/bin/python3

import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run detection')

    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the directory containing images and labels')
    parser.add_argument(
        '--model', type=str, default='U-Net',
        choices=('U-Net', 'SegNet', 'DeepLab', 'FCN'),
        help='Model architecture')
    parser.add_argument(
        '--weights_path', type=str, default=None,
        help='Input weights path')
    parser.add_argument(
        '--visualization_path', type=str, default='/tmp',
        help='Path to a directory where the detection visualizations '
             'will be saved')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The number of samples that will be propagated through the '
             'network at once')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Generator random seed')
    parser.add_argument(
        '--tensor_height', type=int, default=256,
        help='Height of the tensor representing the image')
    parser.add_argument(
        '--tensor_width', type=int, default=256,
        help='Width of the tensor representing the image')
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
        '--validation_set_percentage', type=float, default=0.2,
        help='If generating the dataset - Percentage of the entire dataset to '
             'be used for the detection in the form of a decimal number')
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
    parser.add_argument(
        '--ignore_masks', type=utils.str2bool, default=False,
        help='Boolean to decide if computing also average statstics based on '
             'grand truth data or running only the prediction')

    args = parser.parse_args()

    # check required arguments by individual operations
    if args.weights_path is None:
        raise parser.error(
            'Argument weights_path required')
    if not 0 <= args.validation_set_percentage <= 1:
        raise parser.error(
            'Argument validation_set_percentage must be greater or equal to 0 '
            'and smaller than 1')

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from smoderp2d.detect import run

    run(args.data_dir, args.model, args.weights_path, args.visualization_path,
        args.batch_size, args.seed, (args.tensor_height, args.tensor_width),
        args.force_dataset_generation, args.fit_dataset_in_memory,
        args.validation_set_percentage, args.filter_by_classes,
        args.backbone, args.ignore_masks)
