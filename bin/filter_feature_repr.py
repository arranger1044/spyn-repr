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

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode

from spn.linked.representation import load_feature_info
from spn.linked.representation import store_feature_info
from spn.linked.representation import filter_features_by_layer
from spn.linked.representation import filter_features_by_scope_length
from spn.linked.representation import feature_mask_from_info
from spn.linked.representation import filter_features_by_node_type

from spn.linked.representation import load_features_from_file

from spn.linked.representation import FeatureInfo

from spn.factory import build_theanok_spn_from_block_linked

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
INFO_FILE_EXT = 'features.info'
SCOPE_FILE_EXT = 'scopes'

FMT_DICT = {
    'int': '%d',
    'float': '%.18e',
    'float.8': '%.8e',
}

NODE_TYPE_DICT = {
    'sum': SumNode.__name__,
    'prod': ProductNode.__name__,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str,
                        help='Dataset dir')

    parser.add_argument('-r', '--repr-data', type=str,
                        default=None,
                        help='Learned feature dataset name')

    parser.add_argument('--train-ext', type=str,
                        help='Training set name regex')

    parser.add_argument('--valid-ext', type=str,
                        help='Validation set name regex')

    parser.add_argument('--test-ext', type=str,
                        help='Test set name regex')

    parser.add_argument('--info', type=str,
                        help='Path to feature info file')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./data/repr/',
                        help='Output dir path')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--fmt', type=str, nargs='?',
                        default='float',
                        help='Dataset output number formatter')

    parser.add_argument('--sep', type=str, nargs='?',
                        default=',',
                        help='Dataset output separator')

    parser.add_argument('--dtype', type=str, nargs='?',
                        default='float',
                        help='Loaded dataset type')

    parser.add_argument('--layers', type=int, nargs='+',
                        default=None,
                        help='Layer ids to extract')

    parser.add_argument('--scopes', type=int, nargs='+',
                        default=None,
                        help='Scope lengths to extract')

    parser.add_argument('--nodes', type=str, nargs='+',
                        default=None,
                        help='Node types to extract (sum|prod)')

    parser.add_argument('--no-ext', action='store_true',
                        help='Whether to concatenate the new representation to the old dataset')

    parser.add_argument('--save-text', action='store_true',
                        help='Saving the filtered text to text as well')

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
    dataset_path = args.dataset

    train = None
    valid = None
    test = None

    pickle_split_path = os.path.join(dataset_path, '{}.{}'.format(args.repr_data,
                                                                  PICKLE_SPLIT_EXT))
    print('Looking for {}'.format(pickle_split_path))
    if os.path.exists(pickle_split_path):
        logging.info('Loading from pickle {}'.format(pickle_split_path))
        with open(pickle_split_path, 'rb') as split_file:
            train, valid, test = pickle.load(split_file)

    else:
        splits = []
        for s in [args.train_ext,
                  args.valid_ext,
                  args.test_ext]:
            split_path = os.path.join(dataset_path, '{}.{}'.format(args.repr_data,
                                                                   s))
            splits.append(numpy.loadtxt(split_path, dtype=args.dtype, delimiter=','))
        train, valid, test = splits
        # train, valid, test = dataset.load_dataset_splits(dataset_path,
        #                                                  filter_regex=[args.train_ext,
        #                                                                args.valid_ext,
        #                                                                args.test_ext])
        # dataset_name = args.train_ext.split('.')[0]
    dataset_name = args.repr_data

    n_instances = train.shape[0]
    n_features = train.shape[1]
    assert train.shape[1] == valid.shape[1]
    assert valid.shape[1] == test.shape[1]

    logging.info('\ttrain: {}\n\tvalid: {}\n\ttest: {}'.format(train.shape,
                                                               valid.shape,
                                                               test.shape))

    logging.info('Loading feature info from {}'.format(args.info))
    feat_s_t = perf_counter()
    feature_info = load_feature_info(args.info)
    feat_e_t = perf_counter()
    logging.info('\tdone in {} secs'.format(feat_e_t - feat_s_t))

    assert len(feature_info) == n_features

    filter_prefix = "filtered"

    #
    # filtering by layer?
    if args.layers:

        layer_range = None

        if len(args.layers) == 1:
            layer_range = range(0, args.layers)
        elif len(args.layers) == 2:
            layer_range = range(args.layers[0], args.layers[1])

        filter_prefix += ".l_{}.".format(args.layers)

        logging.info('Filtering by layer (range {})'.format(layer_range))
        filtered_info = []
        for layer in layer_range:
            logging.info('\tgetting layer {}'.format(layer))
            filtered_info.extend(filter_features_by_layer(feature_info, layer))

        feature_info = filtered_info

    #
    # filtering by scope length?
    if args.scopes:

        scope_range = None

        if len(args.scopes) == 1:
            scope_range = range(1, args.scopes)
        elif len(args.scopes) == 2:
            scope_range = range(args.scopes[0], args.scopes[1])

        filter_prefix += ".s_{}.".format(args.scopes)

        logging.info('Filtering by scope length {} (range {})'.format(args.scopes, scope_range))
        filtered_info = []
        for scope_length in scope_range:
            logging.info('\tgetting scope of length {}'.format(scope_length))
            filtered_info.extend(filter_features_by_scope_length(feature_info, scope_length))

        feature_info = filtered_info

    #
    # filtering by node type
    if args.nodes:

        filter_prefix += ".n_{}.".format(args.scopes)

        logging.info('Filtering by node types ({})'.format(args.nodes))
        filtered_info = []
        for node_type in args.nodes:
            logging.info('\tgetting nodes of type {} ({})'.format(node_type,
                                                                  NODE_TYPE_DICT[node_type]))
            filtered_info.extend(filter_features_by_node_type(feature_info,
                                                              NODE_TYPE_DICT[node_type]))

        feature_info = filtered_info

    logging.info('\n')
    logging.info('Remaining features {} -> {}\n'.format(n_features, len(feature_info)))

    #
    # saving to file filtered info
    feature_info_output_path = os.path.join(args.output, '{}.filtered.{}'.format(args.suffix,
                                                                                 INFO_FILE_EXT))
    logging.info('Saving filtered feature info to {}'.format(feature_info_output_path))
    store_feature_info(feature_info, feature_info_output_path)

    #
    # generating the mask
    feature_mask = feature_mask_from_info(feature_info, n_features)

    #
    # applying the mask to the data
    filt_train = train[:, feature_mask]
    filt_valid = valid[:, feature_mask]
    filt_test = test[:, feature_mask]

    assert filt_train.shape[0] == train.shape[0]
    assert filt_valid.shape[0] == valid.shape[0]
    assert filt_test.shape[0] == test.shape[0]

    assert filt_train.shape[1] == filt_valid.shape[1]
    assert filt_valid.shape[1] == filt_test.shape[1]

    logging.info('New shapes:\n\ttrain: {}\n\tvalid: {}\n\ttest: {}'.format(filt_train.shape,
                                                                            filt_valid.shape,
                                                                            filt_test.shape))

    #
    # remapping to new feature order ids, starting from 0
    ordered_feature_info = [FeatureInfo(i,
                                        info.node_id,
                                        info.layer_id,
                                        info.node_type,
                                        info.node_scope)
                            for i, info in enumerate(sorted(feature_info,
                                                            key=lambda x: x.feature_id))]
    feature_info_output_path = os.path.join(args.output, '{}.{}'.format(args.suffix,
                                                                        INFO_FILE_EXT))
    logging.info('Saving remapped feature info to {}'.format(feature_info_output_path))
    store_feature_info(ordered_feature_info, feature_info_output_path)

    #
    # storing them
    if args.save_text:
        train_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix,
                                                                  args.train_ext))
        valid_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix,
                                                                  args.valid_ext))
        test_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix,
                                                                 args.test_ext))

        logging.info('\nSaving training set to: {}'.format(train_out_path))
        numpy.savetxt(train_out_path, filt_train, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving validation set to: {}'.format(valid_out_path))
        numpy.savetxt(valid_out_path, filt_valid, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving test set to: {}'.format(test_out_path))
        numpy.savetxt(test_out_path, filt_test, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

    #
    # saving to pickle
    split_file_path = os.path.join(args.output, '{}.{}'.format(args.suffix,
                                                               PICKLE_SPLIT_EXT))
    logging.info('Saving pickle data splits to: {}'.format(split_file_path))
    with open(split_file_path, 'wb') as split_file:
        pickle.dump((filt_train, filt_valid, filt_test), split_file, protocol=4)
