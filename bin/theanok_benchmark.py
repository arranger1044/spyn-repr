import sys
sys.setrecursionlimit(50000)

import dataset

import numpy
from numpy.testing import assert_array_almost_equal

import theano.misc.pkl_utils

import datetime

import os

import logging

from spn.utils import stats_format
from spn.linked.spn import evaluate_on_dataset

from spn.theanok.spn import *
from spn.theanok.layers import *
from spn.theanok.spn import evaluate_on_dataset_batch
from spn.factory import build_theanok_spn_from_block_linked

import pickle

from time import perf_counter

import argparse

THEANO_MODEL_EXT = 'theano_model'

#
# TODO: make this parametric by cli
# loading spn
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='Specify a dataset name from data/ (es. caltech101)')


parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-m', '--model', type=str,
                    default='models/caltech101/caltech101_spn_500/best.caltech101.model',
                    help='Model path to load')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('--max-nodes-layer', type=int,
                    default=None,
                    help='Max number of nodes per layer')


#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG)

model_path = args.model

logging.info('\nLoading spn model from: {}'.format(model_path))
spn = None
with open(model_path, 'rb') as model_file:
    load_start_t = perf_counter()
    spn = pickle.load(model_file)
    load_end_t = perf_counter()
    logging.info('done in {}'.format(load_end_t - load_start_t))

#
# loading dataset
dataset_name = args.dataset
logging.info('Loading dataset {}'.format(dataset_name))
train, valid, test = dataset.load_train_val_test_csvs(dataset_name, path='data/')

logging.info('\nEvaluating on training set')
eval_s_t = perf_counter()
train_preds = evaluate_on_dataset(spn, train)
eval_e_t = perf_counter()
train_avg_ll = numpy.mean(train_preds)
logging.info('\t{}'.format(train_avg_ll))
logging.info('\tdone in {}'.format(eval_e_t - eval_s_t))

logging.info('Evaluating on validation set')
eval_s_t = perf_counter()
valid_preds = evaluate_on_dataset(spn, valid)
eval_e_t = perf_counter()
valid_avg_ll = numpy.mean(valid_preds)
logging.info('\t{}'.format(valid_avg_ll))
logging.info('\tdone in {}'.format(eval_e_t - eval_s_t))

logging.info('Evaluating on test set')
eval_s_t = perf_counter()
test_preds = evaluate_on_dataset(spn, test)
eval_e_t = perf_counter()
test_avg_ll = numpy.mean(test_preds)
logging.info('\t{}'.format(test_avg_ll))
logging.info('\tdone in {}'.format(eval_e_t - eval_s_t))

freqs, features = dataset.data_2_freqs(train)

logging.info('Encoding train')
ind_train = dataset.one_hot_encoding(train, feature_values=features)
logging.info('\t{}'.format(ind_train.shape))

logging.info('Encoding valid')
ind_valid = dataset.one_hot_encoding(valid, feature_values=features)
logging.info('\t{}'.format(ind_valid.shape))

logging.info('Encoding test')
ind_test = dataset.one_hot_encoding(test, feature_values=features)
logging.info('\t'.format(ind_test.shape))

#
# converting the spn
theano_model_path = '{}.{}'.format(args.model, THEANO_MODEL_EXT)
theano_spn = None
logging.info('Looking for theano spn model in {}'.format(theano_model_path))
if os.path.exists(theano_model_path):
    logging.info('Loading theanok pickle model')
    with open(theano_model_path, 'rb') as mfile:
        # theano_spn = theano.misc.pkl_utils.load(mfile)
        theano_spn = BlockLayeredSpn.load(mfile)
        # theano_spn = pickle.load(mfile)
else:
    logging.info('Creating model anew')
    theano_spn = build_theanok_spn_from_block_linked(spn, ind_train.shape[1], features,
                                                     max_n_nodes_layer=args.max_nodes_layer)
    logging.info('Saving model to {}'.format(theano_model_path))
    with open(theano_model_path, 'wb') as mfile:
        # theano.misc.pkl_utils.dump(theano_spn, mfile)
        theano_spn.dump(mfile)
        # pickle.dump(theano_spn, mfile)

logging.info('Theanok spn:\n{}'.format(theano_spn))

#
# evaluating theanok spn with mini batches
# batch_sizes = [100, 200, 500, 1000, None]
batch_sizes = [None]

for batch_size in batch_sizes:
    logging.info('\n\n\tsize: {}'.format(batch_size))
    logging.info('Evaluating on training set')
    eval_s_t = perf_counter()
    b_train_preds = evaluate_on_dataset_batch(theano_spn, ind_train, batch_size)
    eval_e_t = perf_counter()
    b_train_avg_ll = numpy.mean(b_train_preds)
    logging.info('\t{}'.format(b_train_avg_ll))
    logging.info('\tdone in {}'.format(eval_e_t - eval_s_t))

    logging.info('Evaluating on validation set')
    eval_s_t = perf_counter()
    b_valid_preds = evaluate_on_dataset_batch(theano_spn, ind_valid, batch_size)
    eval_e_t = perf_counter()
    b_valid_avg_ll = numpy.mean(b_valid_preds)
    logging.info('\t{}'.format(b_valid_avg_ll))
    logging.info('\tdone in {}'.format(eval_e_t - eval_s_t))

    logging.info('Evaluating on test set')
    eval_s_t = perf_counter()
    b_test_preds = evaluate_on_dataset_batch(theano_spn, ind_test, batch_size)
    eval_e_t = perf_counter()
    b_test_avg_ll = numpy.mean(b_test_preds)
    logging.info('\t{}'.format(b_test_avg_ll))
    logging.info('\tdone in {}'.format(eval_e_t - eval_s_t))

    assert_array_almost_equal(b_train_preds, train_preds, decimal=5)
    assert_array_almost_equal(b_valid_preds, valid_preds, decimal=5)
    assert_array_almost_equal(b_test_preds, test_preds, decimal=5)
