from spn.linked.representation import load_features_from_file
from spn.linked.representation import save_features_to_file

import numpy
from numpy.testing import assert_array_equal

import os

import logging

import argparse


def batch_feature_split(feature_path, batch_size):
    #
    # load features first
    feature_masks = load_features_from_file(feature_path)
    n_features = len(feature_masks)
    logging.info('Loaded {} feature masks from {}'.format(n_features,
                                                          feature_path))
    feature_splits = []
    for i in range(0, n_features, batch_size):
        masks_split = feature_masks[i:i + batch_size]
        feature_splits.append(masks_split)
        logging.info('Considering range {}:{} (size {})'.format(i, i + batch_size,
                                                                len(masks_split)))

    tot_n_features = sum([len(s) for s in feature_splits])
    logging.info('Prepare serialization for {} splits and {} tot features'.format(len(feature_splits),
                                                                                  tot_n_features))
    assert tot_n_features == n_features

    feature_split_paths = []
    for i, masks in enumerate(feature_splits):
        split_path = '{}.{}.{}'.format(feature_path,
                                       i * batch_size,
                                       min(i * batch_size + batch_size - 1,
                                           n_features - 1))
        save_features_to_file(masks, split_path)
        logging.info('Saved split (size {}) to {}'.format(len(masks), split_path))
        feature_split_paths.append(split_path)

    return feature_masks, feature_split_paths, feature_splits


def load_batch_feature_splits(feature_paths):
    feature_masks = []
    for split_path in feature_paths:
        split_masks = load_features_from_file(split_path)
        print(split_masks.shape)
        feature_masks.append(split_masks)

    return numpy.concatenate(feature_masks, axis=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str,
                        help='Feature file path')

    parser.add_argument('-s', '--split', type=int,
                        help='Batch split size')

    #
    # parsing the args
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting with arguments:\n%s", args)

    #
    # save them
    feature_masks, split_paths, feature_splits = batch_feature_split(args.path, args.split)

    #
    # loading them back as a sanity check
    rec_feature_masks = load_batch_feature_splits(split_paths)

    assert_array_equal(feature_masks, rec_feature_masks)
