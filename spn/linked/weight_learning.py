import numpy

from scipy.misc import logsumexp

import numba

from .nodes import SumNode
from .nodes import ProductNode
from .nodes import CategoricalSmoothedNode
from .nodes import CategoricalIndicatorNode
from .nodes import CLTreeNode

from ..factory import retrieve_children_parent_assoc

from collections import deque

from spn import LOG_ZERO
from spn import MARG_IND

RAND_SEED = 1337


@numba.jit
def evaluate_indicator_node(node, data):
    """
    WRITEME
    """
    n_instances = data.shape[0]
    lls = numpy.zeros(n_instances)

    for i in range(n_instances):
        if data[i, node.var] != node.var_val and data[i, node.var] != MARG_IND:
            lls[i] = LOG_ZERO

    return lls


@numba.jit
def evaluate_categorical_node(node, data):
    """
    WRITEME
    """
    n_instances = data.shape[0]
    lls = numpy.zeros(n_instances)

    for i in range(n_instances):
        obs_val = data[i, node.var]
        if obs_val == MARG_IND:
            lls[i] = 0.
        else:
            lls[i] = node._var_probs[obs_val]

    return lls


@numba.jit
def evaluate_product_node(node,  eval_assoc, n_instances):
    """
    WRITEME
    """
    lls = numpy.zeros(n_instances)

    for child in node.children:
        lls += eval_assoc[child]

    return lls


@numba.jit
def evaluate_sum_node(node, eval_assoc, n_instances):
    """
    WRITEME
    """
    n_children = len(node.children)
    ll_hats = numpy.zeros((n_instances, n_children))
    log_weights = numpy.zeros(n_children)

    for i, (child, log_weight) in enumerate(zip(node.children,
                                                node.log_weights)):
        log_weights[i] = log_weight
        ll_hats[:, i] = eval_assoc[child]

    lls = logsumexp(ll_hats + log_weights[numpy.newaxis, :], axis=1)

    return lls


def evaluate_node(node, data, eval_assoc):
    """
    Dispatching node evaluation by node type
    """

    n_instances = data.shape[0]

    if isinstance(node, CategoricalIndicatorNode):
        return evaluate_indicator_node(node, data)
    elif isinstance(node, CategoricalSmoothedNode):
        return evaluate_categorical_node(node, data)
    elif isinstance(node, SumNode):
        return evaluate_sum_node(node, eval_assoc, n_instances)
    elif isinstance(node, ProductNode):
        return evaluate_product_node(node, eval_assoc, n_instances)


def ml_weights_estimation_posterior(node,
                                    eval_assoc):
    #
    # retrieve all children and their past evaluations
    children_evals = numpy.array([numpy.sum(numpy.exp(eval_assoc[child]))
                                  for child in node.children])
    ml_weights = children_evals / numpy.sum(children_evals)
    return ml_weights


def ml_weights_estimation_posterior_I(node,
                                      eval_assoc):
    #
    # retrieve all children and their past evaluations
    children_evals = numpy.concatenate([numpy.exp(eval_assoc[child])[..., None]
                                        for child in node.children], axis=1)
    ml_weights = children_evals / numpy.sum(children_evals, axis=1)[..., None]
    return ml_weights.mean(axis=0)  # / n_instances


def ml_weights_estimation_posterior_II(node,
                                       eval_assoc):
    #
    # retrieve all children and their past evaluations
    children_evals = numpy.concatenate([numpy.exp(eval_assoc[child])[..., None]
                                        for child in node.children], axis=1)
    children_evals = children_evals * numpy.array(node.weights)[None, :]
    ml_weights = children_evals / numpy.sum(children_evals, axis=1)[..., None]
    return ml_weights.mean(axis=0)  # / n_instances


def ml_weights_estimation_counts(node,
                                 eval_assoc):
    children_evals = [eval_assoc[child] for child in node.children]
    children_evals = numpy.concatenate([e.reshape(e.shape[0], 1)
                                        for e in children_evals],
                                       axis=1)
    children_attr = numpy.argmax(children_evals, axis=1)
    children_counts = numpy.bincount(children_attr,
                                     minlength=children_evals.shape[1])
    ml_weights = children_counts / sum(children_counts)
    return ml_weights


def ml_evaluation(spn,
                  data,
                  nodes_to_eval=None,
                  child_assoc=None,
                  update_weights=False,
                  nodes_to_skip_updating=None,
                  weight_estimator=ml_weights_estimation_posterior_II):
    """
    Estimating the weights in a linked spn by traversing it bottom up
    Nodes_to_eval a set of nodes to evaluate (after being evaluated the algo stops)
    """

    n_instances = data.shape[0]

    if child_assoc is None:
        child_assoc = retrieve_children_parent_assoc(spn)
    else:
        child_assoc = dict(child_assoc)

    #
    #
    if nodes_to_eval is None:
        nodes_to_eval = set()

    if nodes_to_skip_updating is None:
        nodes_to_skip_updating = set()

    nodes_evals = {}
    layer_nodes_evals = {}
    weight_updates = {}

    #
    # remove one node from memory if it has no more parents to evaluate
    def remove_child_parent(parent_node):
        if hasattr(parent_node, 'children') and parent_node.children:
            for child in parent_node.children:
                child_assoc[child].remove(parent_node)
                if not child_assoc[child]:
                    layer_nodes_evals.pop(child)

    #
    # proceeding one layer at a time
    for layer in spn.bottom_up_layers():

        stop = False

        for node in layer.nodes():

            if update_weights:
                #
                # before evaluating, is this a sum node? can we evaluate it?
                if isinstance(node, SumNode) and node in nodes_to_eval:
                    #                                                n_instances)
                    ml_weights = weight_estimator(node, layer_nodes_evals)

                    # node.set_weights(ml_weights)
                    weight_updates[node] = ml_weights

            instances_evals = evaluate_node(node, data, layer_nodes_evals)

            #
            # can we remove its children values from memory?
            remove_child_parent(node)

            layer_nodes_evals[node] = instances_evals

            #
            # if is to eval, store the evaltuation to return it later
            if node in nodes_to_eval:
                nodes_evals[node] = instances_evals
                nodes_to_eval.remove(node)

            #
            # if no more nodes to eval, exit
            if not nodes_to_eval:
                stop = True
                break
        if stop:
            break

    return nodes_evals, weight_updates


def random_weight_estimation(nodes,
                             data,
                             rand_gen=None):
    """
    Setting random weights to sum nodes
    """
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    for node in nodes:
        #
        # checking for correct type
        assert isinstance(node, SumNode)
        #
        # random weights
        rand_weights = rand_gen.rand(len(node.children))
        rand_weights /= numpy.sum(rand_weights)

        node.set_weights(rand_weights)

    return nodes


@numba.jit
def estimate_counts_numba(data,
                          instance_ids,
                          feature_ids,
                          feature_vals,
                          estimated_counts=None):
    """
    Assuming that estimated_counts is a numpy 2D array
    (features x max(feature_val))
    """

    if estimated_counts is None:
        n_features = len(feature_ids)
        max_feature_val = max(feature_vals)
        estimated_counts = numpy.zeros((n_features, max_feature_val))

    #
    # actual counting
    for feature_id in feature_ids:
        for instance_id in instance_ids:
            estimated_counts[feature_id, data[instance_id, feature_id]] += 1

    return estimated_counts


@numba.jit
def smooth_ll_parameters(estimated_counts,
                         ll_frequencies,
                         instance_ids,
                         feature_ids,
                         feature_vals,
                         alpha):
    """
    WRITEME
    """
    tot_counts = len(instance_ids)
    for feature_id in feature_ids:
        feature_val = feature_vals[feature_id]
        smooth_tot_ll = numpy.log(tot_counts + feature_val * alpha)
        for val in range(feature_val):
            smooth_n = estimated_counts[feature_id, val] + alpha
            smooth_n_ll = numpy.log(smooth_n) if smooth_n > 0.0 else LOG_ZERO
            ll_frequencies[feature_id, val] = smooth_n_ll - smooth_tot_ll
    return ll_frequencies


def random_weight_ml_estimation(nodes,
                                data,
                                alpha=0.0,
                                rand_gen=None):
    """
    WRITEME
    """
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    for node in nodes:
        #
        # checking for correct type
        assert isinstance(node, SumNode)
        #
        # random clustering

        #
        # estimate data on each clustering partition


def bootstrap_weight_ml_estimation(nodes,
                                   data,
                                   alpha=0.0,
                                   bootstrap_rate=0.5):
    """
    WRITEME
    """
