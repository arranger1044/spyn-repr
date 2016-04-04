import numpy

import matplotlib
import matplotlib.pyplot as pyplot

import pickle

import os

from scipy.io import loadmat

RANDOM_SEED = 1337


def load_caltech101_from_mat(data_path,
                             split_names=['train',
                                          'val',
                                          'test'],
                             data_suffix='_data',
                             label_suffix='_labels',
                             class_names='classnames'):
    data_dict = loadmat(data_path)
    data_splits = [(data_dict[split + data_suffix], data_dict[split + label_suffix])
                   for split in split_names]
    #
    # un raveling the y
    data_splits = [(split_x, split_y.flatten()) for split_x, split_y in data_splits]
    return data_splits


def save_caltech101_pickle(data_splits, output_path):
    with open(output_path, 'wb') as data_file:
        pickle.dump(data_splits, data_file)


def load_caltech101_pickle(data_path):
    data_splits = None
    with open(data_path, 'rb') as data_file:
        data_splits = pickle.load(data_file)
    return data_splits


def plot_m_by_n_images(images,
                       m, n,
                       fig_size=(12, 12),
                       cmap=matplotlib.cm.binary):
    fig = pyplot.figure(figsize=fig_size)
    for x in range(m):
        for y in range(n):
            ax = fig.add_subplot(m, n, n * y + x + 1)
            ax.matshow(images[n * y + x], cmap=cmap)
            pyplot.xticks(numpy.array([]))
            pyplot.yticks(numpy.array([]))
    pyplot.show()


def array_2_mat(array, n_rows=28):
    return array.reshape(n_rows, -1)


def plot_caltech101_silhouettes(image_arrays,
                                m, n,
                                fig_size=(12, 12),
                                invert=True,
                                cmap=matplotlib.cm.binary):

    image_matrixes = None
    if invert:
        image_matrixes = [array_2_mat(1 - img).T for img in image_arrays]
    else:
        image_matrixes = [array_2_mat(img).T for img in image_arrays]

    plot_m_by_n_images(image_matrixes, m, n, fig_size, cmap)
