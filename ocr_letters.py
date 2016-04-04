import numpy

import matplotlib
import matplotlib.pyplot as pyplot

import pickle

import os


def load_ocr_letters_data_split_from_txt(data_path):

    data = numpy.loadtxt(data_path, delimiter=' ')

    x, y = data[:, :-1].astype(numpy.int32), data[:, -1].astype(numpy.int32)

    print('Loaded dataset:\n\tx: {}\ty: {}'.format(x.shape, y.shape))
    assert x.shape[0] == y.shape[0]
    assert y.ndim == 1
    assert x.shape[1] == 128

    return x, y


def load_ocr_letters_from_txt(data_dir, split_names=['ocr_letters_train.txt',
                                                     'ocr_letters_valid.txt',
                                                     'ocr_letters_test.txt']):
    split_paths = [os.path.join(data_dir, file_name)
                   for file_name in split_names]
    data_splits = [load_ocr_letters_data_split_from_txt(path) for path in split_paths]
    return data_splits


def save_ocr_letters_pickle(data_splits, output_path):
    with open(output_path, 'wb') as data_file:
        pickle.dump(data_splits, data_file)


def load_ocr_letters_pickle(data_path):
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


def array_2_mat(array, n_rows=16):
    return array.reshape(n_rows, -1)


def plot_ocr_letters(image_arrays,
                     m, n,
                     n_rows=16,
                     fig_size=(12, 12),
                     invert=True,
                     cmap=matplotlib.cm.binary):

    image_matrixes = None
    if invert:
        image_matrixes = [array_2_mat(1 - img, n_rows) for img in image_arrays]
    else:
        image_matrixes = [array_2_mat(img, n_rows) for img in image_arrays]

    plot_m_by_n_images(image_matrixes, m, n, fig_size, cmap)
