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

from sklearn import neural_network

from spn.utils import stats_format
from spn import MARG_IND

import pickle

MODEL_EXT = 'model'
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

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./exp/rbm/',
                        help='Output dir path')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--sep', type=str, nargs='?',
                        default=',',
                        help='Dataset output separator')

    parser.add_argument('--fmt', type=str, nargs='?',
                        default='int',
                        help='Dataset output number formatter')

    parser.add_argument('--n-hidden', type=int, nargs='+',
                        default=[500],
                        help='Number of hidden units')

    parser.add_argument('--l-rate', type=float, nargs='+',
                        default=[0.1],
                        help='Learning rate for training')

    parser.add_argument('--batch-size', type=int, nargs='+',
                        default=[10],
                        help='Batch size during learning')

    parser.add_argument('--n-iters', type=int, nargs='+',
                        default=[10],
                        help='Number of epochs')

    parser.add_argument('--no-ext', action='store_true',
                        help='Whether to concatenate the new representation to the old dataset')

    parser.add_argument('--log', action='store_true',
                        help='Transforming the repr data with log')

    parser.add_argument('--save-model', action='store_true',
                        help='Whether to store the model file as a pickle file')

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

    logging.info('Opening log file...')
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.output + dataset_name + '_' + date_string
    out_log_path = out_path + '/exp.log'
    test_lls_path = out_path + '/test.lls'
    os.makedirs(out_path, exist_ok=True)

    repr_train = None
    repr_valid = None
    repr_test = None

    #
    #
    # performing a grid search along the hyperparameter space
    best_valid_avg_pll = -numpy.inf
    best_params = {}
    best_model = None

    n_hidden_values = args.n_hidden
    learning_rate_values = args.l_rate
    batch_size_values = args.batch_size
    n_iter_values = args.n_iters

    preamble = ("""n-hidden:\tlearning-rate:\tbatch-size:\tn-iters:""" +
                """\ttrain_pll\tvalid_pll:\ttest_pll\n""")

    with open(out_log_path, 'w') as out_log:

        out_log.write("parameters:\n{0}\n\n".format(args))
        out_log.write(preamble)
        out_log.flush()
        #
        # looping over all parameters combinations
        for n_hidden in n_hidden_values:
            for l_rate in learning_rate_values:
                for batch_size in batch_size_values:
                    for n_iters in n_iter_values:

                        logging.info('Learning RBM for {} {} {} {}'.format(n_hidden,
                                                                           l_rate,
                                                                           batch_size,
                                                                           n_iters))
                        #
                        # learning
                        rbm = neural_network.BernoulliRBM(n_components=n_hidden,
                                                          learning_rate=l_rate,
                                                          batch_size=batch_size,
                                                          n_iter=n_iters,
                                                          verbose=args.verbose - 1,
                                                          random_state=rand_gen)
                        fit_s_t = perf_counter()
                        rbm.fit(train)
                        fit_e_t = perf_counter()
                        logging.info('Trained in {} secs'.format(fit_e_t - fit_s_t))

                        #
                        # evaluating training
                        eval_s_t = perf_counter()
                        train_plls = rbm.score_samples(train)
                        eval_e_t = perf_counter()
                        train_avg_pll = numpy.mean(train_plls)
                        logging.info('\tTrain avg PLL: {} ({})'.format(train_avg_pll,
                                                                       eval_e_t - eval_s_t))

                        #
                        # evaluating validation
                        eval_s_t = perf_counter()
                        valid_plls = rbm.score_samples(valid)
                        eval_e_t = perf_counter()
                        valid_avg_pll = numpy.mean(valid_plls)
                        logging.info('\tValid avg PLL: {} ({})'.format(valid_avg_pll,
                                                                       eval_e_t - eval_s_t))

                        #
                        # evaluating test
                        eval_s_t = perf_counter()
                        test_plls = rbm.score_samples(test)
                        eval_e_t = perf_counter()
                        test_avg_pll = numpy.mean(test_plls)
                        logging.info('\tTest avg PLL: {} ({})'.format(test_avg_pll,
                                                                      eval_e_t - eval_s_t))

                        #
                        # checking for improvements on validation
                        if valid_avg_pll > best_valid_avg_pll:
                            best_valid_avg_pll = valid_avg_pll
                            best_model = rbm
                            best_params['n-hidden'] = n_hidden
                            best_params['learning-rate'] = l_rate
                            best_params['batch-size'] = batch_size
                            best_params['n-iters'] = n_iters
                            best_test_plls = test_plls

                            #
                            # saving the model
                            if args.save_model:
                                prefix_str = stats_format([n_hidden,
                                                           l_rate,
                                                           batch_size,
                                                           n_iters],
                                                          '_',
                                                          digits=5)
                                model_path = os.path.join(out_path,
                                                          'best.{0}.{1}'.format(dataset_name,
                                                                                MODEL_EXT))
                                with open(model_path, 'wb') as model_file:
                                    pickle.dump(rbm, model_file)
                                    logging.info('Dumped RBM to {}'.format(model_path))

                        #
                        # writing to file a line for the grid
                        stats = stats_format([n_hidden,
                                              l_rate,
                                              batch_size,
                                              n_iters,
                                              train_avg_pll,
                                              valid_avg_pll,
                                              test_avg_pll],
                                             '\t',
                                             digits=5)
                        out_log.write(stats + '\n')
                        out_log.flush()

        #
        # writing as last line the best params
        out_log.write("{0}".format(best_params))
        out_log.flush()

    #
    # saving the best test_lls
    numpy.savetxt(test_lls_path, best_test_plls, delimiter='\n')

    logging.info('Grid search ended.')
    logging.info('Best params:\n\t%s', best_params)

    #
    # now creating the new datasets from best model
    logging.info('\nConverting training set')
    feat_s_t = perf_counter()
    repr_train = best_model.transform(train)
    feat_e_t = perf_counter()
    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

    logging.info('Converting validation set')
    feat_s_t = perf_counter()
    repr_valid = best_model.transform(valid)
    feat_e_t = perf_counter()
    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

    logging.info('Converting test set')
    feat_s_t = perf_counter()
    repr_test = best_model.transform(test)
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
    # log transform as well?
    if args.log:
        log_repr_train = numpy.log(repr_train)
        log_repr_valid = numpy.log(repr_valid)
        log_repr_test = numpy.log(repr_test)

    # extending the original dataset
    ext_train = None
    ext_valid = None
    ext_test = None
    log_ext_train = None
    log_ext_valid = None
    log_ext_test = None

    if args.no_ext:
        ext_train = repr_train
        ext_valid = repr_valid
        ext_test = repr_test

        if args.log:
            log_ext_train = log_repr_train
            log_ext_valid = log_repr_valid
            log_ext_test = log_repr_test

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

        if args.log:
            log_ext_train = numpy.concatenate((train, log_repr_train), axis=1)
            log_ext_valid = numpy.concatenate((valid, log_repr_valid), axis=1)
            log_ext_test = numpy.concatenate((test, log_repr_test), axis=1)

            assert train.shape[0] == log_ext_train.shape[0]
            assert valid.shape[0] == log_ext_valid.shape[0]
            assert test.shape[0] == log_ext_test.shape[0]
            assert ext_train.shape[1] == train.shape[1] + log_repr_train.shape[1]
            assert ext_valid.shape[1] == valid.shape[1] + log_repr_valid.shape[1]
            assert ext_test.shape[1] == test.shape[1] + log_repr_test.shape[1]

    logging.info('New shapes {0} {1} {2}'.format(ext_train.shape,
                                                 ext_valid.shape,
                                                 ext_test.shape))

#
    # storing them
    train_out_path = os.path.join(out_path, '{}.{}'.format(args.suffix, args.train_ext))
    valid_out_path = os.path.join(out_path, '{}.{}'.format(args.suffix, args.valid_ext))
    test_out_path = os.path.join(out_path, '{}.{}'.format(args.suffix, args.test_ext))

    logging.info('\nSaving training set to: {}'.format(train_out_path))
    numpy.savetxt(train_out_path, ext_train, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

    logging.info('Saving validation set to: {}'.format(valid_out_path))
    numpy.savetxt(valid_out_path, ext_valid, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

    logging.info('Saving test set to: {}'.format(test_out_path))
    numpy.savetxt(test_out_path, ext_test, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

    split_file_path = os.path.join(out_path, '{}.{}.{}'.format(args.suffix,
                                                               dataset_name,
                                                               PICKLE_SPLIT_EXT))
    logging.info('Saving pickle data splits to: {}'.format(split_file_path))
    with open(split_file_path, 'wb') as split_file:
        pickle.dump((ext_train, ext_valid, ext_test), split_file)

    if args.log:
        # storing them
        log_train_out_path = os.path.join(out_path,
                                          'log.{}.{}'.format(args.suffix, args.train_ext))
        log_valid_out_path = os.path.join(out_path,
                                          'log.{}.{}'.format(args.suffix, args.valid_ext))
        log_test_out_path = os.path.join(out_path,
                                         'log.{}.{}'.format(args.suffix, args.test_ext))

        logging.info('\nSaving log training set to: {}'.format(log_train_out_path))
        numpy.savetxt(log_train_out_path,
                      log_ext_train, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving log validation set to: {}'.format(log_valid_out_path))
        numpy.savetxt(log_valid_out_path,
                      log_ext_valid, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving log test set to: {}'.format(log_test_out_path))
        numpy.savetxt(log_test_out_path,
                      log_ext_test, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        log_split_file_path = os.path.join(out_path, 'log.{}.{}.{}'.format(args.suffix,
                                                                           dataset_name,
                                                                           PICKLE_SPLIT_EXT))
        logging.info('Saving pickle log data splits to: {}'.format(log_split_file_path))
        with open(log_split_file_path, 'wb') as split_file:
            pickle.dump((log_ext_train, log_ext_valid, log_ext_test), split_file)
