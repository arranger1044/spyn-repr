

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

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import neighbors
from sklearn import decomposition
from sklearn import manifold

from mnist import load_mnist_pickle
from caltech101 import load_caltech101_pickle
from newsgroups import load_20newsgroups_pickle

import pickle

import matplotlib
import matplotlib.pyplot as pyplot

MAX_N_INSTANCES = 10000

PICKLE_SPLIT_EXT = 'pickle'

BMNIST_PATH = 'data/mnist/binary_mnist_splits.pickle'
CALTECH101_PATH = 'data/caltech101/caltech101_silhouettes.pickle'
NEWSGROUPS_PATH = 'data/20newsgroups/20newsgroups_5000.pickle'
OCR_LETTERS_PATH = 'data/ocr_letters/ocr_letters.pickle'

PREPROCESS_DICT = {
    'std-scl': StandardScaler
}

LOGISTIC_MOD_DICT_PARAMS = {
    'l2-ovr-bal': {
        'penalty': 'l2',
        'tol': 0.0001,
        'fit_intercept': True,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'multi_class': 'ovr',
    },
    'l2-ovr-bal-lbfgs': {
        'penalty': 'l2',
        'tol': 0.0001,
        'fit_intercept': True,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        'multi_class': 'ovr',
    },
    'l2-ovr-bal-sag': {
        'penalty': 'l2',
        'tol': 0.0001,
        'fit_intercept': True,
        'class_weight': 'balanced',
        'solver': 'sag',
        'multi_class': 'ovr',
    },
    'l2-mul-bal': {
        'penalty': 'l2',
        'tol': 0.0001,
        'fit_intercept': True,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'multi_class': 'multinomial',
    }}

KNN_DICT_PARAMS = {
    'uni': {'weights': 'uniform',
            # 'algorithm' : 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'metric_params': None,
            }, }

VIS_METHODS_DICT = {
    'tsne': manifold.TSNE,
    'pca': decomposition.PCA}

# VIS_METHODS_PARAMS_DICT = {
#     'tsne': {},
#     'pca':}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Specify a dataset name (es. nltcs)')

    parser.add_argument('-r', '--repr-data', type=str,
                        default=None,
                        help='Learned feature dataset name')

    parser.add_argument('--repr-dir', type=str, nargs='?',
                        default='data/repr/bmnist/',
                        help='Learned feature dir')

    parser.add_argument('--dtype', type=str, nargs='?',
                        default='int32',
                        help='Loaded dataset type')

    parser.add_argument('-s', '--splits', type=str, nargs='+',
                        default=None,
                        help='Splits names')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./exp/learnspn-b/',
                        help='Output dir path')

    # parser.add_argument('-a', '--alpha', type=float, nargs='+',
    #                     default=[0.1],
    #                     help='Smoothing factor for leaf probability estimation')

    parser.add_argument('--visualize', type=str, nargs='+',
                        default=[],
                        help='Algorithms for visualizing the features (es tsne|pca)')

    parser.add_argument('--preprocess', type=str, nargs='+',
                        default=[],
                        help='Algorithms to preprocess data')

    parser.add_argument('--logistic', type=str, nargs='?',
                        default=None,
                        help='parametrized version of the logistic regression')

    parser.add_argument('--log-c', type=float, nargs='+',
                        default=[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
                        help='logistic ')

    parser.add_argument('--knn', type=str, nargs='?',
                        default=None,
                        help='Algorithms to learn')

    parser.add_argument('--knn-k', type=int, nargs='+',
                        default=[1, 3, 5, 11, 25, 51],
                        help='k-NN k values')

    # parser.add_argument('--model', type=str,
    #                     help='Spn model file path')

    parser.add_argument('--feature-inc', type=int, nargs='+',
                        default=None,
                        help='Considering features in batches')

    parser.add_argument('--exp-name', type=str, nargs='?',
                        default=None,
                        help='Experiment name, if not present a date will be used')

    parser.add_argument('--concat', action='store_true',
                        help='Whether to concatenate the new representation to the old dataset')

    parser.add_argument('--save-model', action='store_true',
                        help='Whether to store the model file as a pickle file')

    parser.add_argument('--eval-only-orig', action='store_true',
                        help='Whether to evaluate only the original data')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    #
    # parsing the args
    args = parser.parse_args()

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    #
    # loading the dataset splits
    #
    logging.info('Loading datasets: %s', args.dataset)
    dataset_name = args.dataset

    dataset_splits = None
    if dataset_name == 'bmnist':
        logging.info('Loading bmnist from pickle')
        dataset_splits = load_mnist_pickle(BMNIST_PATH)
    elif dataset_name == 'caltech101':
        logging.info('Loading caltech101-silhouettes from pickle')
        dataset_splits = load_mnist_pickle(CALTECH101_PATH)
    elif dataset_name == '20newsgroups':
        logging.info('Loading 20newsgroups from pickle')
        dataset_splits = load_20newsgroups_pickle(NEWSGROUPS_PATH)
    elif dataset_name == 'ocr_letters':
        logging.info('Loading ocr letters from pickle')
        dataset_splits = load_20newsgroups_pickle(OCR_LETTERS_PATH)
    else:
        dataset_splits = dataset.load_train_val_test_csvs(dataset_name,
                                                          type=args.dtype,
                                                          suffixes=args.splits)
    for i, split in enumerate(dataset_splits):
        logging.info('\tsplit {}, shape {}, labels {}'.format(i,
                                                              split[0].shape,
                                                              split[1].shape))

    #
    # loading the learned representations
    #
    logging.info('Loading repr splits from {}'.format(args.repr_data))
    repr_splits = None
    pickle_split_path = os.path.join(args.repr_dir, '{}.{}'.format(args.repr_data,
                                                                   PICKLE_SPLIT_EXT))

    #
    # Opening the file for test prediction
    #
    if args.exp_name:
        out_path = os.path.join(args.output, dataset_name + '_' + args.exp_name)
    else:
        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(args.output, dataset_name + '_' + date_string)
    out_log_path = os.path.join(out_path, 'exp.log')
    os.makedirs(out_path, exist_ok=True)

    logging.info('Opening log file {}...'.format(out_log_path))

    #
    # shall we concatenate them? or just adding the labels?
    #
    labelled_splits = None
    if args.eval_only_orig:
        logging.info('Classification only on original data')
        labelled_splits = dataset_splits
    else:

        print('Looking for {}'.format(pickle_split_path))
        if os.path.exists(pickle_split_path):
            logging.info('Loading from pickle {}'.format(pickle_split_path))
            with open(pickle_split_path, 'rb') as split_file:
                repr_splits = pickle.load(split_file)
        else:
            repr_splits = []
            for s in args.splits:
                split_path = os.path.join(args.repr_dir, args.repr_data + s)
                repr_splits.append(numpy.loadtxt(split_path, dtype=args.dtype, delimiter=','))
            # repr_splits = dataset.load_train_val_test_csvs(args.repr_data,
            #                                                path=args.repr_dir,
            #                                                type=args.dtype,
            #                                                suffixes=args.splits)

        for i, split in enumerate(repr_splits):
            logging.info('\tsplit {}, shape {}'.format(i, split.shape))

        labelled_splits = []
        for repr_x, (split_x, split_y) in zip(repr_splits, dataset_splits):
            if args.concat:
                new_repr_x = numpy.concatenate((split_x, repr_x), axis=1)
                assert new_repr_x.shape[0] == split_x.shape[0]
                assert new_repr_x.shape[1] == split_x.shape[1] + repr_x.shape[1]
                logging.info('Concatenated representations: {} -> {}'.format(repr_x.shape,
                                                                             new_repr_x.shape))

                labelled_splits.append([new_repr_x, split_y])
            else:
                labelled_splits.append([repr_x, split_y])

    #
    # preprocessing
    if args.preprocess:
        for prep in args.preprocess:
            preprocessor = PREPROCESS_DICT[prep]()
            logging.info('Preprocessing with {}:'.format(preprocessor))
            #
            # assuming the first split is the training set
            preprocessor.fit(labelled_splits[0][0])
            for i in range(len(labelled_splits)):
                labelled_splits[i][0] = preprocessor.transform(labelled_splits[i][0])

    with open(out_log_path, 'w') as out_log:

        out_log.write("parameters:\n{0}\n\n".format(args))
        out_log.flush()

        train_x, train_y = labelled_splits[0]

        #
        # classification task: logistic
        if args.logistic:
            logging.info('Logistic regression')

            if args.feature_inc:
                min_feature = 0
                max_feature = train_x.shape[1]
                increment = None
                if len(args.feature_inc) == 1:
                    increment = args.feature_inc[0]
                elif len(args.feature_inc) == 2:
                    min_feature = args.feature_inc[0]
                    increment = args.feature_inc[1]
                elif len(args.feature_inc) == 3:
                    min_feature = args.feature_inc[0]
                    max_feature = args.feature_inc[1]
                    increment = args.feature_inc[2]
                else:
                    raise ValueError('More than three values specified for --feature-inc')

                for m in range(min_feature + increment, max_feature + 1, increment):
                    #
                    # selecting subset features
                    logging.info('Considering features {}:{}'.format(min_feature, m))
                    sel_labelled_splits = []
                    for i in range(len(labelled_splits)):
                        sel_labelled_splits.append((labelled_splits[i][0][:, min_feature:m],
                                                    labelled_splits[i][1]))

                    for i in range(len(labelled_splits)):
                        logging.info('shapes {} {}'.format(sel_labelled_splits[i][0].shape,
                                                           sel_labelled_splits[i][1].shape))
                    #
                    # reselecting train
                    train_x, train_y = sel_labelled_splits[0]
                    for c in args.log_c:

                        log_res = linear_model.LogisticRegression(C=c,
                                                                  **LOGISTIC_MOD_DICT_PARAMS[args.logistic])
                        #
                        # fitting
                        fit_s_t = perf_counter()
                        log_res.fit(train_x, train_y)
                        fit_e_t = perf_counter()

                        logging.info('\tC: {} ({})'.format(c, fit_e_t - fit_s_t))
                        #
                        # scoring
                        accs = []
                        for split_x, split_y in sel_labelled_splits:
                            split_s_t = perf_counter()
                            split_acc = log_res.score(split_x, split_y)
                            split_e_t = perf_counter()

                            accs.append(split_acc)

                            logging.info('\t\tacc: {} ({})'.format(split_acc,
                                                                   split_e_t - split_s_t))

                        #
                        # saving to file
                        out_log.write('{0}\t{1}\t{2}\n'.format(m,
                                                               c,
                                                               '\t'.join(str(a) for a in accs)))
                        out_log.flush()
            else:

                for c in args.log_c:

                    log_res = linear_model.LogisticRegression(C=c,
                                                              **LOGISTIC_MOD_DICT_PARAMS[args.logistic])
                    #
                    # fitting
                    fit_s_t = perf_counter()
                    log_res.fit(train_x, train_y)
                    fit_e_t = perf_counter()

                    logging.info('C: {} ({})'.format(c, fit_e_t - fit_s_t))
                    #
                    # scoring
                    accs = []
                    for split_x, split_y in labelled_splits:
                        split_s_t = perf_counter()
                        split_acc = log_res.score(split_x, split_y)
                        split_e_t = perf_counter()

                        accs.append(split_acc)

                        logging.info('\tacc: {} ({})'.format(split_acc, split_e_t - split_s_t))

                    #
                    # saving to file
                    out_log.write('{0}\t{1}\n'.format(c, '\t'.join(str(a) for a in accs)))
                    out_log.flush()

        #
        # classification task: k-nn
        if args.knn:
            logging.info('k-Nearest Neighbors')
            for k in args.knn_k:

                knn = neighbors.KNeighborsClassifier(n_neighbors=k,
                                                     **KNN_DICT_PARAMS[args.knn])
                #
                # fitting
                fit_s_t = perf_counter()
                knn.fit(train_x, train_y)
                fit_e_t = perf_counter()

                logging.info('k: {} ({})'.format(k, fit_e_t - fit_s_t))
                #
                # scoring
                accs = []
                for split_x, split_y in labelled_splits:
                    split_s_t = perf_counter()
                    split_acc = knn.score(split_x, split_y)
                    split_e_t = perf_counter()

                    accs.append(split_acc)

                    logging.info('\tacc: {} ({})'.format(split_acc, split_e_t - split_s_t))

                #
                # saving to file
                out_log.write('{0}\t{1}\n'.format(k, '\t'.join(str(a) for a in accs)))
                out_log.flush()

        #
        # visualizing?
        if args.visualize:

            n_components = 2

            vis_x = None
            vis_y = None
            #
            # if train is too large, visualizing the test split portion
            if train_x.shape[0] > MAX_N_INSTANCES:
                logging.info('Visualizing test')
                vis_x, vis_y = labelled_splits[-1]
            else:
                logging.info('Visualizing training')
                vis_x = train_x
                vis_y = train_y

            for decomp in args.visualize:
                #
                # decomposing in 2D
                t0 = perf_counter()
                tsne = VIS_METHODS_DICT[decomp](n_components=n_components)
                dec_vis_x = tsne.fit_transform(vis_x)
                t1 = perf_counter()
                logging.info("Applied decomposition {} in {} sec".format(decomp,
                                                                         t1 - t0))

                #
                # visualize the new components
                fig = pyplot.figure(figsize=(16, 14))
                pyplot.scatter(
                    dec_vis_x[:, 0], dec_vis_x[:, 1], c=vis_y, cmap=matplotlib.cm.Spectral)
                pyplot.show()
