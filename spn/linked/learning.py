from spn.linked.spn import Spn

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalSmoothedNode

from spn.linked.layers import SumLayer
from spn.linked.layers import ProductLayer
from spn.linked.layers import CategoricalSmoothedLayer

from spn.factory import SpnFactory

from spn import RND_SEED

import itertools

import numpy

import scipy
import scipy.spatial.distance

from collections import deque

from sklearn.cluster import spectral_clustering
from sklearn.manifold import spectral_embedding

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import sys

import gc
#
# Some util classes
#


class DataSlice(object):

    """
    A little util class for storing
    the sets of indexes for the instances and features
    considered
    """

    class_counter = 0

    @classmethod
    def reset_id_counter(cls):
        """
        WRITEME
        """
        DataSlice.class_counter = 0

    @classmethod
    def whole_slice(cls,
                    n_instances,
                    n_features):
        # lists can be as good as sets atm
        # instances = {i for i in range(n_instances)}
        # features = {i for i in range(n_features)}
        instances = [i for i in range(n_instances)]
        features = [i for i in range(n_features)]
        return DataSlice(instances, features)

    def __init__(self,
                 instances=None,
                 features=None):
        self.instances = instances
        self.features = features
        self.id = DataSlice.class_counter
        DataSlice.class_counter += 1


class NodeBuild(object):

    """
    WRITEME
    """

    def __init__(self,
                 id,
                 children_ids,
                 children_weights=None):
        self.id = id
        self.children_ids = children_ids
        self.children_weights = children_weights


class SpectralStructureLearner(object):

    """
    WRITEME
    """

    def __init__(self,
                 sigma=0.1,
                 rand_gen=None):
        """
        WRITEME
        """
        self._sigma = sigma

        # initing the random generator
        if rand_gen is not None:
            self._rand_gen = rand_gen
        else:
            self._rand_gen = numpy.random.RandomState(RND_SEED)

    def gaussian_kernel(self, instance_1, instance_2):
        """
        WRITEME - like NG's
        e ^ (-(||s_1-s_2||^2)/2*sigma**2)
        """
        return numpy.exp(- (numpy.linalg.norm(instance_1 - instance_2, 2) ** 2)
                         / (2 * self._sigma ** 2))

    def estimate_counts(feature):
        """
        WRITEME
        """
        # seems like scipy is converting them to float
        # this is potentially highly memory consuming
        # if not done inplace
        feature = feature.astype('int8', copy=False)
        # frequency counting
        feature_count = numpy.bincount(feature)
        # getting the number of different values seen
        feature_vals = feature_count.shape[0]
        # this can be tricky: have to cope with the eventuality
        # of seeing only zeros
        if feature_vals < 2:
            feature_vals = 2
            feature_count = numpy.append(feature_count, [0], 0)

        return feature_count, feature_vals

    def g_test_val(self, instance_1, instance_2):
        """
        WRITEME
        """

        # instance_1 = instance_1.astype('int8', copy=False)
        # instance_2 = instance_2.astype('int8', copy=False)

        n_samples = instance_1.shape[0]
        # print(instance_1, instance_2, n_samples)

        # feature_tot_1 = numpy.bincount(instance_1)
        # feature_tot_2 = numpy.bincount(instance_2)

        # feature_vals_1 = feature_tot_1.shape[0]
        # feature_vals_2 = feature_tot_2.shape[0]

        feature_tot_1, feature_vals_1 = \
            SpectralStructureLearner.estimate_counts(instance_1)
        feature_tot_2, feature_vals_2 = \
            SpectralStructureLearner.estimate_counts(instance_2)

        # print(feature_tot_1,
        #       feature_vals_1,
        #       feature_tot_2,
        #       feature_vals_2)

        # computing the contingency table
        # shall I use numpy.histogram2d?
        # or the pull request numpy.table?
        co_occ_matrix = numpy.zeros((feature_vals_1, feature_vals_2),
                                    dtype='int64')

        for val_1, val_2 in zip(instance_1, instance_2):
            co_occ_matrix[val_1][val_2] += 1
        # print(co_occ_matrix)
        # expected frequencies
        exp_freqs = numpy.outer(feature_tot_1, feature_tot_2) / n_samples
        # print(exp_freqs)
        g_val_matrix = \
            numpy.where(co_occ_matrix > 0,
                        co_occ_matrix * numpy.log(co_occ_matrix / exp_freqs),
                        0.0)
        # g_val_matrix = co_occ_matrix * numpy.log(co_occ_matrix / exp_freqs)

        return g_val_matrix.sum() * 2

    def g_test(self, instance_1, instance_2, p_value):
        """
        Applying a G-test
        """
        # extracting counts
        # TODO: re-eng this in a cleaner way: g_test_val does this
        # computation as well
        feature_tot_1, feature_vals_1 = \
            SpectralStructureLearner.estimate_counts(instance_1)
        feature_tot_2, feature_vals_2 = \
            SpectralStructureLearner.estimate_counts(instance_2)

        # computing the deegres of freedon
        feature_nonzero_1 = numpy.count_nonzero(feature_tot_1)
        feature_nonzero_2 = numpy.count_nonzero(feature_tot_2)

        dof = (feature_nonzero_1 - 1) * (feature_nonzero_2 - 1)

        # computing the G val
        g_val = self.g_test_val(instance_1, instance_2)

        # print('GSTATS', g_val, dof, p_value,  2 * dof * p_value + 0.001)

        # testing against p value
        return g_val < 2 * dof * p_value + 0.001

    def compute_similarity_matrix_pair(self,
                                       data_slice,
                                       metric=gaussian_kernel):
        """
        From a matrix m x n creates a kernel matrix
        according to a metric of size m x m
        (it shall be symmetric, and (semidefinite) positive)

        ** MANUAL **

        """

        n_instances = data_slice.shape[0]
        print('data slice with {0} instances'.format(n_instances))
        # allocating the matrix
        similarity_matrix = numpy.zeros((n_instances, n_instances))

        # caching the slices
        instances = [instance for instance in data_slice]

        # computing the metric pairwise, just once
        for i, j in itertools.combinations(range(n_instances), 2):
            sys.stdout.write('\rsimilarity between entities {0}-{1}'.
                             format(i, j))
            sys.stdout.flush()
            # from index tuples to instances
            # instance_slice_1 = data_slice[i, :]
            # instance_slice_2 = data_slice[j, :]
            instance_slice_1 = instances[i]
            instance_slice_2 = instances[j]

            # computing the metric on them
            sim_i_j = metric(instance_slice_1, instance_slice_2)
            # filling the matrix
            similarity_matrix[i, j] = sim_i_j
            similarity_matrix[j, i] = sim_i_j
        # then for each value with itself
        for i, instance_slice in enumerate(data_slice):
            sys.stdout.write('\rsimilarity between entities {0}-{1}'.
                             format(i, i))
            sys.stdout.flush()
            sim_i_i = metric(instance_slice,
                             instance_slice)
            similarity_matrix[i, i] = sim_i_i

        print('\n')
        return similarity_matrix

    def compute_similarity_matrix(self,
                                  data_slice,
                                  metric='gaussian'):
        """
        From a matrix m x n creates a kernel matrix
        according to a metric of size m x m
        (it shall be symmetric, and (semidefinite) positive)

        ** USES SCIPY **

        """

        if metric == 'gaussian':
            pairwise_dists = \
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(data_slice,
                                                 'sqeuclidean'))
            similarity_matrix = scipy.exp(-pairwise_dists /
                                          (2 * self._sigma ** 2))
        elif metric == 'gtest':
            similarity_matrix = \
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(data_slice,
                                                 self.g_test_val))

        return similarity_matrix

    #
    # Gens' variants for splitting/clustering
    #
    def greedy_split_features(self,
                              data_slice,
                              slice_ids,
                              g_factor,
                              rand_gen):
        """
        WRITEME

        """

        # assuming the features are here the rows of data_slice
        # equivalently len(feature_ids)
        n_features = data_slice.shape[0]

        # copying for manipulating it
        feature_ids = [i for i in range(n_features)]
        # print('FEATS', slice_ids, feature_ids, n_features)

        # the split will be binary
        dependent_features = []

        # extracting one feature at random
        # this can be done more efficiently with a set TODO
        rand_feature_id = rand_gen.randint(0, n_features)
        feature_ids.remove(rand_feature_id)
        dependent_features.append(slice_ids[rand_feature_id])
        # print('REM', feature_ids)

        # greedy bfs searching
        features_to_process = deque()
        features_to_process.append(rand_feature_id)

        while features_to_process:
            # get one
            current_feature_id = features_to_process.popleft()
            # print('curr FT', current_feature_id)

            # features to remove later
            features_to_remove = deque()

            for other_feature_id in feature_ids:
                # extract the feature slices
                feature_1 = data_slice[current_feature_id, :]
                feature_2 = data_slice[other_feature_id, :]
                # print('---->', current_feature_id,
                # other_feature_id)
                # print(feature_1, feature_2)

                # print('G_VAL', self.g_test_val(feature_1,
                #                                feature_2))

                # apply a G-test
                if not self.g_test(feature_1, feature_2, g_factor):
                    # print('GTEST')
                    features_to_remove.append(other_feature_id)
                    dependent_features.append(slice_ids[other_feature_id])
                    features_to_process.append(other_feature_id)

            # now removing
            # even now a set would be much more efficient
            for feature_id in features_to_remove:
                feature_ids.remove(feature_id)

        # translating remaining features
        other_features = [slice_ids[feature_id] for feature_id in feature_ids]
        clustering = [dependent_features, other_features]
        return clustering

    #
    # Cut related methods
    # TODO: make them inner functions
    #

    def diag_sum(self, W):
        """
        WRITEME
        """
        # D = numpy.zeros(W.shape)
        # for i, row in enumerate(W):
        #     D[i, i] = row.sum()
        # return D

        return numpy.diag(numpy.sum(W, axis=1))

    def cut_val_f(self, W, f):
        """
        W adiacency matrix
        f cluster assignement
        """
        # compute diagonal matrix D
        D = self.diag_sum(W)
        print(D)
        print(D - W)
        print(numpy.dot(D - W, f))
        return numpy.dot(f.T, numpy.dot(D - W, f))

    def cut_val_w(self, W, f):
        """
        W adiacency matrix
        f cluster assignement
        """
        # compute diagonal matrix D
        W_f = W[f == 1, :]
        W_sliced = W_f[:, f == -1]
        return numpy.sum(W_sliced)

    def vol(self, W, f, index):
        """
        WRITEME
        """
        return numpy.sum(W[f == index, :])

    def f_clu(self, W, f):
        vol_f = self.vol(W, f, 1)
        vol_not_f = self.vol(W, f, -1)
        return numpy.where(f == 1, f / vol_f, f / vol_not_f)

    def ncut_val(self, W, f):
        """
        W adiacency matrix
        f cluster assignement
        returns the normalized cut
        """
        # compute diagonal matrix D
        D = self.diag_sum(W)
        return (numpy.dot(f.T, numpy.dot(D - W, f)) /
                numpy.dot(f.T, numpy.dot(D, f)))

    def ncut(self, W, f):
        """
        simpler version
        """
        cut = self.cut_val_w(W, f)
        print('CUT', cut)
        vol_f = self.vol(W, f, 1)
        vol_not_f = self.vol(W, f, -1)
        return (cut * (1. / vol_f + 1. / vol_not_f))

    #
    # clustering management
    #
    def f_assignment_from_clusters(self, W, clustering):
        """
        from [[ids_clu_1], ..., [ids_clu_2]] to f
        """
        f = self.f_from_clusters(clustering)

        # computing normalization by the volume vals of W
        return self.f_clu(W, f)

    def f_from_clusters(self, clustering):
        """
        from [[ids_clu_1], ..., [ids_clu_2]] to f
        """
        # Assuming the clustering to be binary
        # TODO: make it more general
        cluster_1 = clustering[0]
        cluster_not_1 = clustering[1]

        # allocate the f assignment
        f = numpy.ones(len(cluster_1) + len(cluster_not_1))

        # setting now the -1s
        # f[list(cluster_not_1)] = -1
        f[cluster_not_1] = -1

        return f

    def f_from_labels(self, labels):
        """
        from [[ids_clu_1], ..., [ids_clu_2]] to f
        """
        # Assuming the clustering to be binary
        # so labels is a vector of zeros and ones
        # TODO: make it more general

        # allocate the f assignment
        f = numpy.ones(len(labels))

        # setting now the -1s
        # f[list(cluster_not_1)] = -1
        f[labels == 1] = -1

        return f

    def from_labels_to_clustering(self, labels, ids):
        """
        WRITEME
        """
        clustering = {}
        for label, id in zip(labels, ids):
            if label in clustering:
                # clustering[label].add(i)
                clustering[label].append(id)
            else:
                # adding a new cluster
                # clustering[label] = {i}
                clustering[label] = [id]
        return list(clustering.values())

    #
    # verifying clustering quality
    #
    def is_clustering_valid_ncut(self,
                                 clustering,
                                 W=None,
                                 threshold=0.3):
        """
        Computing the NCUT by hand
        """
        valid = False
        if W is not None:
            # computing the ncut

            f = self.f_from_labels(clustering)

            ncut = self.ncut(W, f)
            # test it against the threshold
            if ncut < threshold:
                valid = True
                print('NCUT: {0} RHO: {1} -> valid'.format(ncut, threshold))
            else:
                print('NCUT: {0} RHO: {1} -> not valid'.
                      format(ncut, threshold))
        else:
            # here goes Gens Test TODO
            raise NotImplemented('You have to implement Gens\'!')
        return valid

    def is_clustering_valid_gtest(self,
                                  data_slice,
                                  clustering,
                                  W=None,
                                  threshold=10):
        """
        For the partitioning into A and B in W, each w_ij shall pass the gtest
        """
        valid = False
        if W is not None:
            # always assuming just two partitions
            # extract weights cut
            W_cut = []
            # get the dof for the features
            A_set = [i for i, label in enumerate(clustering) if label == 0]
            B_set = [i for i, label in enumerate(clustering) if label == 1]
            DOF_cut = []
            instances = [instance for instance in data_slice]
            for i, j in zip(A_set, B_set):
                feature_tot_1, feature_vals_1 = \
                    SpectralStructureLearner.estimate_counts(instances[i])
                feature_tot_2, feature_vals_2 = \
                    SpectralStructureLearner.estimate_counts(instances[j])

                # computing the deegres of freedon
                feature_nonzero_1 = numpy.count_nonzero(feature_tot_1)
                feature_nonzero_2 = numpy.count_nonzero(feature_tot_2)

                DOF_cut.append(
                    (feature_nonzero_1 - 1) * (feature_nonzero_2 - 1))
                W_cut.append(W[i, j])

            W_cut = numpy.array(W_cut)
            G_test = 2 * numpy.array(DOF_cut) * threshold + 0.001
            if numpy.all(W_cut < G_test):
                valid = True
                print('W_cut: {0} G_test: {1} -> valid'.format(W_cut, G_test))
            else:
                print('W_cut: {0} G_test: {1} -> not valid'.
                      format(W_cut, G_test))

        else:
            # here goes Gens Test TODO
            raise NotImplemented('You have to implement Gens\'!')
        return valid

    def spectral_clustering(self,
                            data_slice,
                            ids,
                            k_components,
                            affinity_metric='gaussian',
                            cluster_method=None,
                            norm_lap=False,
                            validity_check=False,
                            threshold=None,
                            pair=False,
                            rand_gen=None):
        """
        WRITEME
        """

        if rand_gen is None:
            rand_gen = self._rand_gen

        #
        # create the affinity matrix first
        #
        print('Computing affinity matrix for measure', affinity_metric)
        aff_start_t = perf_counter()
        if pair:
            affinity_metric_func = None
            if affinity_metric == 'gaussian':
                affinity_metric_func = self.gaussian_kernel
            else:
                affinity_metric_func = self.g_test_val
            affinity_matrix = \
                self.compute_similarity_matrix_pair(data_slice,
                                                    affinity_metric_func)
        else:
            affinity_matrix = \
                self.compute_similarity_matrix(data_slice,
                                               affinity_metric)
        aff_end_t = perf_counter()

        print('Affinity metric computed! (in {0} secs)'.
              format(aff_end_t - aff_start_t))
        print(affinity_matrix)

        n_instances = data_slice.shape[0]
        labels = None
        clustering = None
        valid = None

        #
        # garbage collecting
        #
        gc.collect()

        #
        # checking for the rank of the square matrix
        # cannot calculate eigenvectors where #instances = #clusters
        #
        if n_instances == k_components:
            # the split can be easily computed
            spec_start_t = perf_counter()
            labels = [i for i in range(n_instances)]
            print('One cluster for each element ({0} == {1})'
                  .format(k_components, n_instances))
            spec_end_t = perf_counter()

        else:

            #
            # if no cluster method is specified, use sklearn default
            # sklearn.cluster.SpectralClustering (which uses kmeans)
            #
            if cluster_method is None:
                print('Directly using sklearn.cluster.spectral_clustering')
                spec_start_t = perf_counter()
                labels = spectral_clustering(affinity=affinity_matrix,
                                             # ?
                                             n_clusters=k_components,
                                             n_components=k_components,
                                             eigen_solver='lobpcg',
                                             # affinity='precomputed',
                                             random_state=rand_gen)
                spec_end_t = perf_counter()

            else:

                #
                # projection in the eigen space
                #
                spec_start_t = perf_counter()
                eigen_start_t = perf_counter()
                eigen_data_slice = \
                    spectral_embedding(adiacency=affinity_matrix,
                                       n_components=k_components,
                                       # pyamg?
                                       eigen_solver='lobpcg',
                                       random_state=rand_gen,
                                       norm_laplacian=norm_lap)
                eigen_end_t = perf_counter()
                print('Embedded in the eigenspace in {0}'.
                      format(eigen_end_t - eigen_start_t))

                #
                # apply the choosen clustering method now
                #
                labels = cluster_method(eigen_data_slice)
                spec_end_t = perf_counter()

        print('labels', labels)
        print('ids', ids)
        clustering = self.from_labels_to_clustering(labels, ids)
        print('clustering', clustering)

        print('Clustered {0} objects into {1} clusters'
              ' (in {2} secs)'.
              format(n_instances,
                     len(clustering),
                     (spec_end_t - spec_start_t)))

        if validity_check:
            print('Checking for validity')
            valid = self.is_clustering_valid_ncut(labels,  # clustering,
                                                  affinity_matrix,
                                                  threshold=threshold)
            valid = self.is_clustering_valid_gtest(data_slice,
                                                   labels,
                                                   affinity_matrix,
                                                   threshold=9)

        return labels, clustering, valid

    def fit_structure(self,
                      data,
                      feature_sizes,
                      rho=0.9,
                      k_col_clusters=2,
                      k_row_clusters=100,
                      sigma=3.0,
                      min_instances_slice=50,
                      alpha=0.1,
                      pairwise=False):
        """
        Gens +  Spectral
        """

        #
        # resetting global default parameters
        #
        self._sigma = sigma

        print('Starting top down structure learning ...')
        print('\trho:{0}'.format(rho))
        print('\tsigma:{0}'.format(sigma))
        print('\tmin-ins:{0}'.format(min_instances_slice))

        # a queue containing the data slices to process
        slices_to_process = deque()

        # a stack for building nodes
        node_build_stack = deque()

        # a dict to keep track of id->nodes
        node_id_assoc = {}

        #
        tot_num_instances = data.shape[0]
        tot_num_features = data.shape[1]

        # creating the first slice
        whole_slice = DataSlice.whole_slice(tot_num_instances,
                                            tot_num_features)
        slices_to_process.append(whole_slice)

        # keeping track of leaves
        # input_nodes = []

        first_run = True

        #
        # debug stats
        #
        n_nodes = 0
        n_edges = 0
        n_weights = 0

        spn_start_t = perf_counter()

        #
        # iteratively process & split slices
        #
        while slices_to_process:

            # process a slice
            current_slice = slices_to_process.popleft()

            # pointers to the current data slice
            current_instances = current_slice.instances
            current_features = current_slice.features
            current_id = current_slice.id
            n_instances = len(current_instances)
            n_features = len(current_features)
            print('current_id', current_id)
            print('current_instances', current_instances)
            print('current_features', current_features)
            print('n_instances', n_instances)
            print('n_features', n_features)

            # is there a leaf node or we can split?
            if n_features == 1:

                print('---> Just one feature, adding a leaf')
                (feature_id, ) = current_features
                feature_size = feature_sizes[feature_id]

                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                # create the node
                leaf_node = \
                    CategoricalSmoothedNode(var=feature_id,
                                            var_values=feature_size,
                                            data=current_slice_data,
                                            instances=current_instances)
                # store links
                # input_nodes.append(leaf_node)
                leaf_node.id = current_id
                node_id_assoc[current_id] = leaf_node
                print('Created Smooth Node', leaf_node)

                n_nodes += 1

            elif (n_instances <= min_instances_slice and n_features > 1):
                # splitting the slice on each feature
                print('---> Few instances ({0}), splitting on all features'.
                      format(n_instances))
                # child_slices = [DataSlice(current_instances, {feature_id})
                #                 for feature_id in current_features]
                child_slices = [DataSlice(current_instances, [feature_id])
                                for feature_id in current_features]
                slices_to_process.extend(child_slices)
                # for feature_id in current_features:
                # create new slice
                #     child_slice = DataSlice(current_instances, {feature_id})
                # adding it to be processed
                #     slices_to_process.append(child_slice)

                children_ids = [child.id for child in child_slices]
                # the building node is a product one
                build_prod = NodeBuild(current_id,
                                       children_ids)
                node_build_stack.append(build_prod)

                # creating the product node
                prod_node = ProductNode(var_scope=frozenset(current_features))
                prod_node.id = current_id

                node_id_assoc[current_id] = prod_node
                print('Created Prod Node', prod_node)
                print('children', children_ids)

                n_nodes += 1
            else:

                valid_col_split = None

                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                # first run is a split on rows
                if first_run:
                    print('FIRST RUN')
                    first_run = False
                    valid_col_split = False
                else:
                    #
                    # try clustering on cols
                    #
                    labels, clustering, valid_col_split = \
                        self.spectral_clustering(current_slice_data.T,
                                                 ids=current_features,
                                                 k_components=k_col_clusters,
                                                 affinity_metric='gtest',
                                                 cluster_method=None,
                                                 validity_check=True,
                                                 threshold=rho,
                                                 norm_lap=False,
                                                 pair=pairwise)

                #
                # testing how good the clustering on features is
                if valid_col_split:
                    # clustering on columns
                    print('---> Splitting on features')

                    # computing the remaining features
                    # always assuming a binary split
                    dependent_features = clustering[0]
                    other_features = clustering[1]

                    # creating two new data slices
                    first_slice = DataSlice(current_instances,
                                            dependent_features)
                    second_slice = DataSlice(current_instances,
                                             other_features)
                    slices_to_process.append(first_slice)
                    slices_to_process.append(second_slice)

                    children_ids = [first_slice.id, second_slice.id]

                    # building and storing a product node
                    build_prod = NodeBuild(current_id,
                                           children_ids)
                    node_build_stack.append(build_prod)

                    prod_node = \
                        ProductNode(var_scope=frozenset(current_features))
                    prod_node.id = current_id
                    node_id_assoc[current_id] = prod_node
                    print('Created Prod Node', prod_node)
                    print('children', children_ids)

                    n_nodes += 1

                else:
                    # clustering on rows
                    print('---> Splitting on rows')

                    # at most n_rows clusters
                    k_row_clusters = min(k_row_clusters,
                                         n_instances - 1)

                    # sklearn's
                    labels, clustering, _valid = \
                        self.spectral_clustering(current_slice_data,
                                                 ids=current_instances,
                                                 k_components=k_row_clusters,
                                                 affinity_metric='gaussian',
                                                 cluster_method=None,
                                                 norm_lap=False,
                                                 pair=pairwise)
                    # splitting
                    cluster_slices = [DataSlice(cluster, current_features)
                                      for cluster in clustering]
                    cluster_slices_ids = [slice.id
                                          for slice in cluster_slices]
                    cluster_weights = [len(slice.instances) / n_instances
                                       for slice in cluster_slices]

                    # appending for processing
                    slices_to_process.extend(cluster_slices)

                    # building a sum node
                    build_sum = NodeBuild(current_id,
                                          cluster_slices_ids,
                                          cluster_weights)
                    node_build_stack.append(build_sum)

                    sum_node = SumNode(var_scope=frozenset(current_features))
                    sum_node.id = current_id
                    node_id_assoc[current_id] = sum_node
                    print('Created Sum Node', sum_node)
                    print('children', cluster_slices_ids)

                    n_nodes += 1

        #
        # linking the spn graph (parent -> children)
        #
        print('===> Building tree')

        # saving a reference now to the root (the first node)
        root_build_node = node_build_stack[0]
        root_node = node_id_assoc[root_build_node.id]
        print('ROOT', root_node, type(root_node))

        # traversing the building stack
        # to link and prune nodes
        for build_node in reversed(node_build_stack):
            # for build_node in node_build_stack:

            # current node
            current_id = build_node.id
            print('BID', current_id)
            current_children_ids = build_node.children_ids
            current_children_weights = build_node.children_weights

            # retrieving corresponding node
            node = node_id_assoc[current_id]
            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):
                # getting children
                for child_id, child_weight in zip(current_children_ids,
                                                  current_children_weights):
                    child_node = node_id_assoc[child_id]

                    # checking children types as well
                    if isinstance(child_node, SumNode):
                        # this shall be pruned
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            node.add_child(grand_child,
                                           grand_child_weight *
                                           child_weight)

                    else:
                        node.add_child(child_node, child_weight)

            elif isinstance(node, ProductNode):
                # linking children
                for child_id in current_children_ids:
                    child_node = node_id_assoc[child_id]
                    if isinstance(child_node, CategoricalSmoothedNode):
                        pass
                        # print('SMOOTH CHILD')
                        # checking for alternating type
                    if isinstance(child_node, ProductNode):
                        # this shall be pruned
                        for grand_child in child_node.children:
                            node.add_child(grand_child)
                    else:
                        node.add_child(child_node)
                        # print('ADDED SMOOTH CHILD')

        #
        # building layers
        #
        print('===> Layering spn')
        spn = SpnFactory.layered_linked_spn(root_node)

        spn_end_t = perf_counter()
        print('Spn learnt in {0} secs'.format(spn_end_t - spn_start_t))

        print(spn.stats())

        return spn


#
#
#
#
#
#
class CoClusterSlice(object):

    """
    WRITEME
    """

    def __init__(self, row_id, col_id, data_slice):
        self.cc_row_id = row_id
        self.cc_col_id = col_id
        self.data_slice = data_slice


class ClusterH(object):

    """
    WRITEME
    """

    def __init__(self, id, elements=None):
        self.id = id
        self.elements = elements
        self.children = []

    def __repr__(self):
        return "ClusterH: [id: {0} elements: {1} children: {2}]\n".\
            format(self.id,
                   self.elements,
                   [child.id for child in self.children])


class CoClusteringStructureLearner(object):

    """
    WRITEME
    """

    def __init__(self):
        """
        WRITEME
        """

    def read_hierarchy_from_file(self, filename, sep=' '):
        """
        Reads in a file in the format of hicc
        """
        cluster_assignments = [[int(clust_id)
                                for clust_id in line.strip().split(sep)]
                               for line in open(filename, 'r')]
        return cluster_assignments

    def build_linked_hierarchy(self, clusters):
        """
        From the list of list of integer representation
        to a linked one
        """
        # cluster_h_level = {id: ClusterLevel}
        # preallocing a vector of cluster levels
        n_levels = len(clusters) + 1
        levels = [None for i in range(n_levels)]

        # assuming the assignment to be non empty
        n_elements = len(clusters[0])
        cluster_id = 0

        # creating the first level of the hierarchy
        root_cluster = ClusterH(id=cluster_id,
                                elements=[i for i in range(n_elements)])
        cluster_id += 1

        # building a map to store id-> cluster levels
        first_level_map = {root_cluster.id: root_cluster}
        i = 0
        levels[i] = first_level_map

        # for each level
        for i in range(n_levels - 1):

            # get previous level
            previous_level = levels[i]

            # get current assignment
            level_assign = clusters[i]

            # storing this level assoc
            current_level = {}

            # create the clusters for this level
            ord_clusters = set()
            cls_id_buffer = [None for i in range(n_elements)]

            print('i:', i)
            for j in range(n_elements):

                cls_id = level_assign[j]
                # has this cluster been already seen?
                if cls_id not in ord_clusters:

                    print('j:', j)
                    ord_clusters.add(cls_id)
                    new_cls = ClusterH(id=cluster_id,
                                       elements=[k
                                                 for k in range(j, n_elements)
                                                 if level_assign[k] == cls_id])
                    cluster_id += 1
                    print('adding', new_cls.elements, 'to', j)

                    # linking to parent
                    parent_cls = None
                    prev_i = i - 1
                    # not on the first run
                    if prev_i >= 0:
                        previous_assign = clusters[prev_i]
                        parent_cls = previous_level[previous_assign[j]]
                        print('not first run', previous_assign[j])
                    else:
                        parent_cls = root_cluster

                    # checking if parent is the same cluster as child
                    if parent_cls.elements != new_cls.elements:
                        parent_cls.children.append(new_cls)
                        cls_id_buffer[cls_id] = new_cls.id
                        current_level[new_cls.id] = new_cls
                        print('adding child', new_cls.id, 'to', parent_cls.id)
                    else:
                        print('same cluster')
                        cls_id_buffer[cls_id] = parent_cls.id
                        current_level[parent_cls.id] = parent_cls

            # update this level ids
            for k in range(n_elements):
                level_assign[k] = cls_id_buffer[level_assign[k]]

            print('lvl assign', level_assign)

            levels[i + 1] = current_level

        return levels

    def split_into_univariate_dist(self,
                                   data_slice,
                                   node,
                                   node_id_assoc,
                                   feature_sizes,
                                   data):
        """
        WRITEME
        """
        feature_ids = data_slice.features
        instance_ids = data_slice.instances

        # for each feature
        for feature_id in feature_ids:
            # create a new slice
            new_slice = DataSlice(instance_ids, [feature_id])
            feature_size = feature_sizes[feature_id]
            # create a single node after slicing
            data_instance_slice = data[instance_ids, :]
            feature_slice = data_instance_slice[:, [feature_id]]
            leaf_node = CategoricalSmoothedNode(var=feature_id,
                                                var_values=feature_size,
                                                data=feature_slice)
            # storing for later
            leaf_node.id = new_slice.id
            node_id_assoc[leaf_node.id] = leaf_node

            # linking to parent
            if isinstance(node, SumNode):
                # linking with a fake weight, later this shall be rebuild
                node.add_child(leaf_node, 1.0)
            elif isinstance(node, ProductNode):
                node.add_child(leaf_node)

            print('adding a smooth node {0} to node {1} w/f {2}'.
                  format(leaf_node.id, node.id, feature_id))

    def split_by_row(self,
                     co_clusters_to_slices,
                     curr_clusters,
                     node_id_assoc,
                     min_instances_slice,
                     feature_sizes,
                     data):
        """
        WRITEME
        """
        n_slices = len(co_clusters_to_slices)
        # traversing the queue for a fixed length, then adding it
        for i in range(n_slices):
            # getting the first slice
            curr_cc_to_slice = co_clusters_to_slices.popleft()

            curr_slice = curr_cc_to_slice.data_slice
            instance_ids = curr_slice.instances
            feature_ids = curr_slice.features
            col_id = curr_cc_to_slice.cc_col_id

            print('processing slice', curr_slice.id)

            # retrieving corresponding node, it is a sum node
            sum_node = node_id_assoc[curr_slice.id]

            n_instances_slice = len(instance_ids)
            # check if few instances are left in the slice
            if n_instances_slice <= min_instances_slice:
                print('split into univariate distribution')
                self.split_into_univariate_dist(curr_slice,
                                                sum_node,
                                                node_id_assoc,
                                                feature_sizes,
                                                data)
            else:
                # get the corresponding row co_cluster
                row_cc = curr_clusters[curr_cc_to_slice.cc_row_id]

                print('row children n', len(row_cc.children))
                # for each split
                for cc_child in row_cc.children:
                    # copy the instances
                    new_instance_ids = cc_child.elements[:]

                    # splitting the data slice
                    instance_slice = DataSlice(new_instance_ids, feature_ids)

                    # updating the cc
                    new_cc_to_slice = CoClusterSlice(row_id=cc_child.id,
                                                     col_id=col_id,
                                                     data_slice=instance_slice)

                    # adding a product node as child
                    prod_node = ProductNode(var_scope=frozenset(feature_ids))

                    # storing it
                    prod_node.id = instance_slice.id
                    node_id_assoc[prod_node.id] = prod_node

                    # linking to parent
                    node_weight = (1.0 * len(new_instance_ids) /
                                   n_instances_slice)
                    sum_node.add_child(prod_node,
                                       node_weight)

                    print('adding prod node {0} to sum {1} w/w {2}'.
                          format(prod_node.id, sum_node.id, node_weight))
                    # enqueue the slice
                    co_clusters_to_slices.append(new_cc_to_slice)

    def split_by_column(self,
                        co_clusters_to_slices,
                        curr_clusters,
                        node_id_assoc,
                        feature_sizes,
                        data):
        """
        WRITEME
        """
        n_slices = len(co_clusters_to_slices)

        print('assoc in', len(node_id_assoc))

        # traversing the queue for a fixed length
        for i in range(n_slices):

            # get the first slice
            curr_cc_to_slice = co_clusters_to_slices.popleft()

            # getting the data slice
            curr_slice = curr_cc_to_slice.data_slice
            instance_ids = curr_slice.instances
            feature_ids = curr_slice.features

            print('processing slice', curr_slice.id)

            # retieving the corresponding stored node, a product one
            prod_node = node_id_assoc[curr_slice.id]

            # more than one feature, checking for the cocluster splits
            row_id = curr_cc_to_slice.cc_row_id

            # getting the associated column co-cluster
            col_cc = curr_clusters[curr_cc_to_slice.cc_col_id]

            # for each split
            for cc_child in col_cc.children:
                # copying children ids
                new_feature_ids = cc_child.elements[:]

                # splitting the data slice
                feature_slice = DataSlice(instance_ids, new_feature_ids)

                # check for a univariate split
                if len(new_feature_ids) == 1:

                    # adding a smoothed ndoe as leaf
                    self.split_into_univariate_dist(feature_slice,
                                                    prod_node,
                                                    node_id_assoc,
                                                    feature_sizes,
                                                    data)

                else:
                    # updating the co cluster
                    new_cc_to_slice = CoClusterSlice(row_id=row_id,
                                                     col_id=cc_child.id,
                                                     data_slice=feature_slice)

                    # creating corresponding sum node
                    sum_node = SumNode(var_scope=frozenset(new_feature_ids))

                    # properly setting it and storing
                    sum_node.id = feature_slice.id
                    node_id_assoc[sum_node.id] = sum_node

                    # linking to the parent ndoe
                    prod_node.add_child(sum_node)

                    print('adding sum node {0} to prod node {1}'.
                          format(sum_node.id, prod_node.id))

                    # putting the new co cluster enqueue
                    co_clusters_to_slices.append(new_cc_to_slice)

    def build_spn_from_co_clusters(self,
                                   row_hierarchy,
                                   col_hierarchy,
                                   data,
                                   feature_sizes,
                                   min_instances_slice=50,
                                   max_depth=10):
        """
        WRITEME
        """
        n_instances = data.shape[0]
        n_features = data.shape[1]

        # get the number of common levels
        # hierarchies can have different depths
        n_common_levels = min(len(row_hierarchy), len(col_hierarchy))

        # limiting the max depth in the construction
        # shall this prevent overfitting?
        depth = min(max_depth, n_common_levels)

        # creating the first slice
        whole_slice = DataSlice.whole_slice(n_instances, n_features)

        # setting its associated co-clusters
        cc_to_first_slice = CoClusterSlice(row_id=0,
                                           col_id=0,
                                           data_slice=whole_slice)

        # creating the queue to process the cc_slices
        curr_cc_to_slice = deque()
        # and initing with the first one
        curr_cc_to_slice.append(cc_to_first_slice)

        # build the map to store the id->node info
        node_id_assoc = {}

        # the first node is a sum node by convention
        whole_scope = [i for i in range(n_features)]
        root_node = SumNode(var_scope=frozenset(whole_scope))
        root_node.id = whole_slice.id
        node_id_assoc[root_node.id] = root_node

        #
        # Building the spn fron hierarchies up to a certain depth
        #
        for i in range(depth):

            # get the right levels in the hierarchies
            row_clusters = row_hierarchy[i]
            col_clusters = col_hierarchy[i]

            # split by row first
            print('---> Building rows')
            self.split_by_row(curr_cc_to_slice,
                              row_clusters,
                              node_id_assoc,
                              min_instances_slice,
                              feature_sizes,
                              data)

            # then by columns
            print('---> Building cols')
            self.split_by_column(curr_cc_to_slice,
                                 col_clusters,
                                 node_id_assoc,
                                 feature_sizes,
                                 data)

        # the remaining slices are being splitted
        # into univariate leaves (there is no point in considering
        # the remaining levels of one of the hierarchies since we
        # are assuming a perfectly alternated sum/prod levels )
        print('Splitting remaining slices')
        while curr_cc_to_slice:
            # get the cc_slice from the front of the queue
            rem_cc_slice = curr_cc_to_slice.popleft()

            # then the data slice from it
            rem_slice = rem_cc_slice.data_slice

            rem_node = node_id_assoc[rem_slice.id]
            self.split_into_univariate_dist(rem_slice,
                                            rem_node,
                                            node_id_assoc,
                                            feature_sizes,
                                            data)

        # now traversing the linked tree top down
        # with the aim to prune correctly the sum nodes parents of the leaves
        print('Relinking leaves')
        nodes_to_process = deque()
        nodes_to_process.append(root_node)

        while nodes_to_process:
            #
            curr_node = nodes_to_process.popleft()
            #
            # here I am assuming that if a sum node is to be processed
            # then it is valid (pruning comes in action when considering
            # product nodes' sum node children)
            if isinstance(curr_node, SumNode):
                for child in curr_node.children:
                    nodes_to_process.append(child)

            elif isinstance(curr_node, ProductNode):
                # storing references to children to move
                children_to_remove = []
                children_to_add = []

                for child in curr_node.children:
                    if isinstance(child, SumNode):
                        insert = True
                        for grand_child in child.children:
                            # this is not flexible about types
                            if isinstance(grand_child,
                                          CategoricalSmoothedNode):
                                insert = False
                                children_to_add.append(grand_child)
                        if not insert:
                            # marking the child to be removed
                            print('removing sum node')
                            children_to_remove.append(child)
                # removing the children
                curr_node.children = [child for child in curr_node.children
                                      if child not in children_to_remove]
                # adding the new ones
                curr_node.children.extend(children_to_add)

                # equeuing the children
                nodes_to_process.extend(curr_node.children)

        #
        # from the linked representation to a linked and layered one
        #
        spn = SpnFactory.layered_linked_spn(root_node)

        return spn
