import subprocess

import numpy

import os

import argparse

import logging

import datetime

import re

try:
    from time import perf_counter
except:
    from time import time as perf_counter


MTLEARN_EXEC = './mtlearn'
MSCORE_EXEC = './mscore'
SPN2AC_EXEC = './spn2ac'


SPN_EXT = '.spn'
AC_EXT = '.ac'
DATA_DIR = 'data/'
TRAIN_EXT = '.ts.data'
VALID_EXT = '.valid.data'
TEST_EXT = '.test.data'


def ll_array_from_model_score(score_output):
    """
    Quick and dirty parsing
    """
    #
    # split strings by newlines
    lines = score_output.split('\n')
    #
    # remove all the lines that are not numbers
    lls = []
    for ll in lines:
        try:
            lls.append(float(ll))
        except ValueError:
            pass
    #
    # convert to numpy array
    return numpy.array(lls)


def model_score(model, dataset, exec_path=MSCORE_EXEC, single_instances=False):
    """
    Computing the LL of the model on a dataset using mscore
    """
    process = None
    if single_instances:
        process = subprocess.Popen([exec_path,
                                    '-m', model,
                                    '-i', dataset,
                                    '-v'],
                                   stdout=subprocess.PIPE)
    else:
        process = subprocess.Popen([exec_path,
                                    '-m', model,
                                    '-i', dataset],
                                   stdout=subprocess.PIPE)
    proc_out, proc_err = process.communicate()

    #
    # TODO manage errors
    # print(proc_out)
    if proc_err is not None:
        print('Mscore Errors:')
        print(proc_err)
    # else:
    #    print(proc_out)

    #
    # parsing the output
    if single_instances:
        return ll_array_from_model_score(proc_out.decode("utf-8"))  # this shall be completed
    else:
        avg_ll, std_ll = re.findall(b"[-+]?\d*\.\d+|\d+", proc_out)
        return avg_ll, std_ll


def adding_bins_to_path(bin_dir):
    """
    """
    path = os.getenv('PATH')
    print(path)
    os.environ["PATH"] = path + ':' + bin_dir
    print(os.getenv('PATH'))


def convert_spn_to_ac(spn_model_path, spn_2_ac_exec_path=SPN2AC_EXEC):
    """
    Converting an spn model (.spn folder) into ACs with spn2ac
    """

    #
    # computing the output file path
    ac_model_path = spn_model_path.replace(SPN_EXT, AC_EXT)
    process = subprocess.Popen([spn_2_ac_exec_path,
                                '-m', spn_model_path,
                                '-o', ac_model_path],
                               stdout=subprocess.PIPE)
    proc_out, proc_err = process.communicate()
    print(proc_out)
    print(proc_err)

    #
    # TODO: manage errors
    return ac_model_path, (proc_out, proc_err)


def stats_format(stats_list, separator, digits=5):
    formatted = []
    float_format = '{0:.' + str(digits) + 'f}'
    for stat in stats_list:
        # if isinstance(stat, int):
        #     formatted.append(str(stat))
        # el
        if isinstance(stat, float):

            formatted.append(float_format.format(stat))
        else:
            formatted.append(str(stat))
    # concatenation
    return separator.join(formatted)


def mtlearn_wrapper(model_path,
                    dataset_path,
                    n_components,
                    seed,
                    exec_path=MTLEARN_EXEC,
                    spn_2_ac_exec_path=SPN2AC_EXEC):
    """
    Wrapping mtlearn executable in python
    """
    mtlearn_start_t = perf_counter()
    #
    # opening a pipe to execute mtlearn
    process = subprocess.Popen([exec_path,
                                '-i', dataset_path,
                                '-o', model_path,
                                '-k', str(n_components),
                                '-seed', str(seed)],
                               stdout=subprocess.PIPE)
    proc_out, proc_err = process.communicate()
    mtlearn_end_t = perf_counter()
    logging.info('Model learned in %f secs', (mtlearn_end_t - mtlearn_start_t))

    #
    # joping not for errors
    if proc_err is not None:
        logging.info('ERRORS: %s', proc_err)

    #
    # mtlearns outputs .spn files, so we need to convert them
    ac_path, conv_out = convert_spn_to_ac(model_path, spn_2_ac_exec_path)

    logging.info('Converted model to AC!\n%s', conv_out[0])

    return ac_path

#
# the main script with argparse
if __name__ == '__main__':

    # bin_dir = '/home/valerio/Petto Redigi/libra-tk-1.0.1/bin'
    # bin_dir = '/root/Desktop/libra_exp/bin/'

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs=1,
                        help='Specify a dataset name from data/ (es. nltcs)')

    parser.add_argument('-n', '--n-components', type=int, nargs='+',
                        default=[2, 50, 2],
                        help='min max inc')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-e', '--exec-path', type=str, nargs='?',
                        default='/home/valerio/Petto Redigi/libra-tk-1.0.1/bin',
                        help='Output dir path')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='exp/mtlearn/',
                        help='Output dir path')

    parser.add_argument('-i', '--n-iters', type=int, nargs='?',
                        default=5,
                        help='Number of trials')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    #
    # adding to the path
    adding_bins_to_path(args.exec_path)

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    seed = args.seed
    MAX_RAND_SEED = 99999999  # sys.maxsize
    numpy_rand_gen = numpy.random.RandomState(seed)

    logging.info('Opening log file...')
    (dataset_name,) = args.dataset
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.output + dataset_name + '_' + date_string
    print('OUTPATH', out_path)
    out_log_path = out_path + '/exp.log'
    test_lls_log_path = out_path + '/test.lls'
    model_subdir_path = out_path + '/models/'
    mean_lls_path = out_path + '/mean.lls'
    best_lls_path = out_path + '/best.lls'

    #
    # getting the paths
    train_path = DATA_DIR + dataset_name + TRAIN_EXT
    valid_path = DATA_DIR + dataset_name + VALID_EXT
    test_path = DATA_DIR + dataset_name + TEST_EXT

    #
    # parsing the components
    min_components = 1
    max_components = None
    increment = 1
    if len(args.n_components) > 3 or len(args.n_components) < 1:
        raise ValueError('More than three values for components')
    elif len(args.n_components) == 3:
        min_components = args.n_components[0]
        max_components = args.n_components[1]
        increment = args.n_components[2]
    elif len(args.n_components) == 2:
        min_components = args.n_components[0]
        max_components = args.n_components[1]
    elif len(args.n_components) == 1:
        max_components = args.n_components[0]

    n_components = (max_components - min_components) // increment + 1
    logging.info('N Components: %d', n_components)

    #
    # creating dir if non-existant
    if not os.path.exists(os.path.dirname(out_log_path)):
        os.makedirs(os.path.dirname(out_log_path))
    if not os.path.exists(os.path.dirname(model_subdir_path)):
        os.makedirs(os.path.dirname(model_subdir_path))

    #
    # keeping track of results
    best_state = {}
    best_state['valid_ll'] = -numpy.Inf
    mean_state = {}

    best_train_lls = numpy.zeros(n_components)
    best_valid_lls = numpy.zeros(n_components)
    best_test_lls = numpy.zeros(n_components)

    best_train_lls.fill(-numpy.Inf)
    best_valid_lls.fill(-numpy.Inf)
    best_test_lls.fill(-numpy.Inf)

    mean_train_lls = numpy.zeros(n_components)
    mean_valid_lls = numpy.zeros(n_components)
    mean_test_lls = numpy.zeros(n_components)

    preamble = ("""n-compo:\t#trial:\tseed:""" +
                """\ttrain_ll\tvalid_ll:\ttest_ll\n""")

    mtlearn_exec_path = os.path.join(args.exec_path, 'mtlearn')
    mscore_exec_path = os.path.join(args.exec_path, 'mscore')
    spn2ac_exec_path = os.path.join(args.exec_path, 'spn2ac')

    with open(out_log_path, 'w') as out_log:

        out_log.write("parameters:\n{0}\n\n".format(args))
        out_log.write(preamble)
        out_log.flush()

        #
        # the main cycle here is on the component
        for j, m in enumerate(range(min_components - 1,
                                    max_components,
                                    increment)):

            seeds = numpy_rand_gen.randint(MAX_RAND_SEED, size=args.n_iters)

            mean_state[m] = {}
            #
            # then we repeat it for a number of trials
            for i in range(args.n_iters):
                logging.info('\n## Repeating trial %d/%d##', i + 1, args.n_iters)
                #
                # compositing the paths
                model_path = model_subdir_path + dataset_name + '_' + \
                    str(m) + '_' + str(i) + SPN_EXT

                print(train_path)
                print(model_path)
                #
                # learning the component
                mt = mtlearn_wrapper(model_path,
                                     train_path,
                                     n_components=m + 1,
                                     exec_path=mtlearn_exec_path,
                                     spn_2_ac_exec_path=spn2ac_exec_path,
                                     seed=seeds[i])
                #
                # evaluating it
                train_avg_ll, _train_std_ll = model_score(mt, train_path,
                                                          exec_path=mscore_exec_path)
                logging.info('TRAIN SET: %s', train_avg_ll)

                valid_avg_ll, _valid_std_ll = model_score(mt, valid_path,
                                                          exec_path=mscore_exec_path)
                logging.info('VALID SET: %s', valid_avg_ll)

                test_lls = model_score(mt, test_path, single_instances=True,
                                       exec_path=mscore_exec_path)
                test_avg_ll = test_lls.mean()
                logging.info('TEST SET: %f', test_avg_ll)

                train_avg_ll = float(train_avg_ll)
                valid_avg_ll = float(valid_avg_ll)

                if valid_avg_ll > best_state['valid_ll']:
                    best_state['valid_ll'] = valid_avg_ll
                    best_state['train_ll'] = train_avg_ll
                    best_state['test_ll'] = test_avg_ll
                    best_state['n_mix'] = m + 1
                    #
                    # saving to file
                    numpy.savetxt(test_lls_log_path, test_lls, delimiter='\n')

                #
                # updating the best stats
                if train_avg_ll > best_train_lls[j]:
                    best_train_lls[j] = float(train_avg_ll)
                if valid_avg_ll > best_valid_lls[j]:
                    best_valid_lls[j] = valid_avg_ll
                if test_avg_ll > best_test_lls[j]:
                    best_test_lls[j] = test_avg_ll

                #
                # and the mean ones
                mean_train_lls[j] += train_avg_ll
                mean_valid_lls[j] += valid_avg_ll
                mean_test_lls[j] += test_avg_ll

                #
                # saving to general log file
                stats = stats_format([m,
                                      i,
                                      seeds[i],
                                      train_avg_ll, valid_avg_ll, test_avg_ll],
                                     '\t', digits=5)
                out_log.write(stats + '\n')
                out_log.flush()

        #
        # writing as last line the best params
        out_log.write("{0}".format(best_state))
        out_log.flush()

    #
    # saving aggregatet stats
    numpy.savetxt(mean_lls_path,
                  numpy.vstack((mean_train_lls / args.n_iters,
                                mean_valid_lls / args.n_iters,
                                mean_test_lls / args.n_iters)),
                  delimiter=',',
                  fmt='%.8e')

    numpy.savetxt(best_lls_path,
                  numpy.vstack((best_train_lls,
                                best_valid_lls,
                                best_test_lls)),
                  delimiter=',',
                  fmt='%.8e')

    logging.info('Exp search ended.')
    logging.info('Best params:\n\t%s', best_state)
