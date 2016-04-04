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

from spn.utils import stats_format
from spn import MARG_IND

from spn.linked.representation import extract_features_marginalization_acquery
from spn.linked.representation import extract_features_marginalization_acquery_opt_unique
from spn.linked.representation import load_features_from_file

import pickle

PREDS_EXT = 'lls'

TRAIN_PREDS_EXT = 'train.{}'.format(PREDS_EXT)
VALID_PREDS_EXT = 'valid.{}'.format(PREDS_EXT)
TEST_PREDS_EXT = 'test.{}'.format(PREDS_EXT)

DATA_EXT = 'data'
TRAIN_DATA_EXT = 'train.{}'.format(DATA_EXT)
VALID_DATA_EXT = 'valid.{}'.format(DATA_EXT)
TEST_DATA_EXT = 'test.{}'.format(DATA_EXT)

PICKLE_SPLIT_EXT = 'pickle'
FEATURE_FILE_EXT = 'features'
SCOPE_FILE_EXT = 'scopes'

FMT_DICT = {'int': '%d',
            'float': '%.18e',
            'float.8': '%.8e',
            }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str,
                        help='Dataset dir')

    parser.add_argument('--train-ext', type=str,
                        help='Training set name regex')

    parser.add_argument('--valid-ext', type=str,
                        help='Validation set name regex')

    parser.add_argument('--test-ext', type=str,
                        help='Test set name regex')

    parser.add_argument('--model', type=str,
                        help='Libra model file path (ac)')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./data/repr/',
                        help='Output dir path')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--sep', type=str, nargs='?',
                        default=',',
                        help='Dataset output separator')

    parser.add_argument('--acquery-path', type=str, nargs='?',
                        default='/home/valerio/Petto Redigi/libra-tk-1.0.1/bin/acquery',
                        help='Path to Libra\'s acquery bin')

    parser.add_argument('--fmt', type=str, nargs='?',
                        default='int',
                        help='Dataset output number formatter')

    parser.add_argument('--features', type=str, nargs='?',
                        default=None,
                        help='Loading feature masks from file')

    parser.add_argument('--no-ext', action='store_true',
                        help='Whether to concatenate the new representation to the old dataset')

    parser.add_argument('--save-text', action='store_true',
                        help='Saving the repr data to text as well')

    parser.add_argument('--overwrite', type=int, nargs='?',
                        default=1,
                        help='Whether to overwrite the generated feature files')

    parser.add_argument('--opt-unique', action='store_true',
                        help='Whether to activate the unique patches opt while computing marg features')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    #
    # parsing the args
    args = parser.parse_args()

    overwrite = True if args.overwrite > 0 else False

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
    dataset_path = args.dataset
    train, valid, test = dataset.load_dataset_splits(dataset_path,
                                                     filter_regex=[args.train_ext,
                                                                   args.valid_ext,
                                                                   args.test_ext])
    dataset_name = args.train_ext.split('.')[0]

    n_instances = train.shape[0]
    n_test_instances = test.shape[0]
    logging.info('\ttrain: {}\n\tvalid: {}\n\ttest: {}'.format(train.shape,
                                                               valid.shape,
                                                               test.shape))
    freqs, feature_vals = dataset.data_2_freqs(train)

    repr_train = None
    repr_valid = None
    repr_test = None

    feature_file_path = args.features
    feature_masks = load_features_from_file(feature_file_path)
    logging.info('Loaded {} feature masks from {}'.format(len(feature_masks),
                                                          feature_file_path))

    extract_feature_func = None
    if args.opt_unique:
        logging.info('Using unique opt')
        extract_feature_func = extract_features_marginalization_acquery_opt_unique
    else:
        extract_feature_func = extract_features_marginalization_acquery

    logging.info('\nConverting training set')
    feat_s_t = perf_counter()
    repr_train = extract_feature_func(train,
                                      args.model,
                                      feature_masks,
                                      args.output,
                                      dtype=float,
                                      prefix=args.suffix,
                                      overwrite_feature_file=overwrite,
                                      exec_path=args.acquery_path)
    feat_e_t = perf_counter()
    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

    logging.info('Converting validation set')
    feat_s_t = perf_counter()
    repr_valid = extract_feature_func(valid,
                                      args.model,
                                      feature_masks,
                                      args.output,
                                      dtype=float,
                                      prefix=args.suffix,
                                      overwrite_feature_file=overwrite,
                                      exec_path=args.acquery_path)
    feat_e_t = perf_counter()
    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

    logging.info('Converting test set')
    feat_s_t = perf_counter()
    repr_test = extract_feature_func(test,
                                     args.model,
                                     feature_masks,
                                     args.output,
                                     dtype=float,
                                     prefix=args.suffix,
                                     overwrite_feature_file=overwrite,
                                     exec_path=args.acquery_path)

    feat_e_t = perf_counter()
    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

    assert train.shape[0] == repr_train.shape[0]
    assert valid.shape[0] == repr_valid.shape[0]
    assert test.shape[0] == repr_test.shape[0]
    logging.info('New shapes {0} {1} {2}'.format(repr_train.shape,
                                                 repr_valid.shape,
                                                 repr_test.shape))

    assert repr_train.shape[1] == repr_valid.shape[1]
    assert repr_valid.shape[1] == repr_test.shape[1]


#
    # extending the original dataset
    ext_train = None
    ext_valid = None
    ext_test = None

    if args.no_ext:
        ext_train = repr_train
        ext_valid = repr_valid
        ext_test = repr_test

    else:
        logging.info('\nConcatenating datasets')
        ext_train = numpy.concatenate((train, repr_train), axis=1)
        ext_valid = numpy.concatenate((valid, repr_valid), axis=1)
        ext_test = numpy.concatenate((test, repr_test), axis=1)

        assert train.shape[0] == ext_train.shape[0]
        assert valid.shape[0] == ext_valid.shape[0]
        assert test.shape[0] == ext_test.shape[0]
        assert ext_train.shape[1] == train.shape[1] + repr_train.shape[1]
        assert ext_valid.shape[1] == valid.shape[1] + repr_valid.shape[1]
        assert ext_test.shape[1] == test.shape[1] + repr_test.shape[1]

    logging.info('New shapes {0} {1} {2}'.format(ext_train.shape,
                                                 ext_valid.shape,
                                                 ext_test.shape))

    #
    # storing them
    if args.save_text:
        train_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix, args.train_ext))
        valid_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix, args.valid_ext))
        test_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix, args.test_ext))

        logging.info('\nSaving training set to: {}'.format(train_out_path))
        numpy.savetxt(train_out_path, ext_train, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving validation set to: {}'.format(valid_out_path))
        numpy.savetxt(valid_out_path, ext_valid, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving test set to: {}'.format(test_out_path))
        numpy.savetxt(test_out_path, ext_test, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

    split_file_path = os.path.join(args.output, '{}.{}.{}'.format(args.suffix,
                                                                  dataset_name,
                                                                  PICKLE_SPLIT_EXT))
    logging.info('Saving pickle data splits to: {}'.format(split_file_path))
    with open(split_file_path, 'wb') as split_file:
        pickle.dump((ext_train, ext_valid, ext_test), split_file)
