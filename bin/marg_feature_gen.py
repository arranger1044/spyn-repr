import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset

import numpy

import datetime

import os

import logging

from spn import MARG_IND

from spn.linked.representation import extract_features_marginalization_rand
from spn.linked.representation import extract_features_marginalization_rectangles

PICKLE_SPLIT_EXT = 'pickle'
FEATURE_FILE_EXT = 'features'
SCOPE_FILE_EXT = 'scopes'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str,
                        help='Dataset dir')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./data/repr/',
                        help='Output dir path')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--rand-marg-rect', type=int, nargs='+',
                        default=None,
                        help='Generating features by marginalization over random rectangles')

    parser.add_argument('--rand-marg', type=int, nargs='+',
                        default=None,
                        help='Generating features by marginalization over random subsets')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    #
    # parsing the args
    args = parser.parse_args()

    #
    # fixing a seed
    rand_gen = numpy.random.RandomState(args.seed)

    os.makedirs(args.output, exist_ok=True)
#
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    #
    # loading dataset splits
    logging.info('Loading datasets: %s', args.dataset)
    dataset_name = args.dataset
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
    logging.info('train shape: {}\nvalid shape: {}\ntest shape: {}'.format(train.shape,
                                                                           valid.shape,
                                                                           test.shape))

    n_instances = train.shape[0]
    n_features = train.shape[1]
    assert valid.shape[1] == n_features
    assert test.shape[1] == n_features

    feature_file_path = '{}.{}.{}'.format(args.suffix,
                                          dataset_name,
                                          FEATURE_FILE_EXT)
    feature_file_path = os.path.join(args.output, feature_file_path)
    logging.info('Saving features to {}'.format(feature_file_path))

    if args.rand_marg:
        logging.info('Rand mask feature generation')
        n_configs = len(args.rand_marg) // 2

        assert len(args.rand_marg) % 2 == 0

        feature_sizes = [args.rand_marg[i * 2] for i in range(n_configs)]
        n_rand_sizes = [args.rand_marg[i * 2 + 1] for i in range(n_configs)]

        logging.info('Features sizes {} rand sizes {}'.format(feature_sizes, n_rand_sizes))

        feat_s_t = perf_counter()

        feature_masks = extract_features_marginalization_rand(n_features,
                                                              feature_sizes,
                                                              n_rand_sizes,
                                                              feature_file_path=feature_file_path,
                                                              marg_value=MARG_IND,
                                                              rand_gen=rand_gen,
                                                              dtype=float)
        feat_e_t = perf_counter()
        logging.info('\tExtracted features in {}'.format(feat_e_t - feat_s_t))

    elif args.rand_marg_rect:
        logging.info('Rand rectangular mask feature generation')
        n_configs = len(args.rand_marg_rect) // 5

        assert len(args.rand_marg_rect) % 5 == 0

        feature_sizes = [args.rand_marg_rect[i * 5] for i in range(n_configs)]
        rect_min_sizes = [(args.rand_marg_rect[i * 5 + 1], args.rand_marg_rect[i * 5 + 2])
                          for i in range(n_configs)]
        rect_max_sizes = [(args.rand_marg_rect[i * 5 + 3], args.rand_marg_rect[i * 5 + 3])
                          for i in range(n_configs)]

        logging.info('Features sizes {} min sizes {} max sizes {}'.format(feature_sizes,
                                                                          rect_min_sizes,
                                                                          rect_max_sizes))
        #
        # a very dirty way to get these parameters
        if 'bmnist' in dataset_name:
            n_rows = 28
            n_cols = 28
        elif 'caltech101' in dataset_name:
            n_rows = 28
            n_cols = 28
        elif 'ocr_letters' in dataset_name:
            n_rows = 16
            n_cols = 8
        else:
            raise ValueError(
                'Unrecognized dataset, cannot retrieve number of columns and rows')

        logging.info('Rectangular stats: features {} rows {} cols {}'.format(n_features,
                                                                             n_rows,
                                                                             n_cols))

        feat_s_t = perf_counter()
        feature_masks = extract_features_marginalization_rectangles(n_features,
                                                                    n_rows, n_cols,
                                                                    feature_sizes,
                                                                    rect_min_sizes,
                                                                    rect_max_sizes,
                                                                    feature_file_path=feature_file_path,
                                                                    marg_value=MARG_IND,
                                                                    rand_gen=rand_gen,
                                                                    dtype=float)
        feat_e_t = perf_counter()
        logging.info('\tExtracted features in {}'.format(feat_e_t - feat_s_t))
