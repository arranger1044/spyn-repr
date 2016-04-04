import numpy
from numpy.testing import assert_array_almost_equal

from ..weight_learning import evaluate_indicator_node
from ..weight_learning import evaluate_categorical_node
from ..weight_learning import evaluate_sum_node
from ..weight_learning import evaluate_product_node
from ..weight_learning import ml_evaluation

from ..nodes import SumNode
from ..nodes import ProductNode
from ..nodes import CategoricalIndicatorNode
from ..nodes import CategoricalSmoothedNode

from ..layers import SumLayer
from ..layers import ProductLayer
from ..layers import CategoricalIndicatorLayer
from ..layers import CategoricalInputLayer


from ..spn import Spn

from spn import LOG_ZERO
from spn import MARG_IND

data = numpy.array([[1, 0, 1, 1, 1, 0],
                    [1, 0, 2, 1, 0, 1],
                    [0, 0, 2, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 1],
                    [0, 0, 2, 0, 0, 0],
                    [1, 0, 1, 1, 1, 1]])
n_instances = data.shape[0]
n_features = data.shape[1]
feature_vals = [2, 2, 3, 2, 2, 2]


def test_evaluate_indicator_node():
    ind_node = CategoricalIndicatorNode(var=0, var_val=1)
    node_eval = evaluate_indicator_node(ind_node, data)
    print(node_eval)
    assert len(node_eval) == n_instances
    node_true_eval_1 = numpy.array([0., 0., LOG_ZERO, LOG_ZERO, 0., 0., LOG_ZERO, 0.])
    assert_array_almost_equal(node_eval, node_true_eval_1)

    ind_node = CategoricalIndicatorNode(var=2, var_val=2)
    node_eval = evaluate_indicator_node(ind_node, data)
    print(node_eval)
    assert len(node_eval) == n_instances
    node_true_eval_2 = numpy.array([LOG_ZERO, 0., 0., LOG_ZERO, LOG_ZERO, LOG_ZERO, 0., LOG_ZERO])
    assert_array_almost_equal(node_eval, node_true_eval_2)


def test_evaluate_categorical_node():
    var = 1
    cat_node = CategoricalSmoothedNode(var=var,
                                       var_values=2,
                                       alpha=0.0,
                                       data=data[:, var][:, numpy.newaxis])
    node_eval = evaluate_categorical_node(cat_node, data)
    print(node_eval)
    assert len(node_eval) == n_instances
    assert_array_almost_equal(node_eval, numpy.zeros(n_instances))

    var = 0
    cat_node = CategoricalSmoothedNode(var=var,
                                       var_values=2,
                                       alpha=0.0,
                                       data=data[:, var][:, numpy.newaxis])
    node_eval = evaluate_categorical_node(cat_node, data)
    print(node_eval)
    assert len(node_eval) == n_instances
    one_prob_val = numpy.sum(data[:, var]) / n_instances
    zero_prob_val = 1 - one_prob_val
    prob_vals = [one_prob_val,
                 one_prob_val,
                 zero_prob_val,
                 zero_prob_val,
                 one_prob_val,
                 one_prob_val,
                 zero_prob_val,
                 one_prob_val]
    assert_array_almost_equal(node_eval, numpy.log(prob_vals))


def test_evaluate_product_node():

    child_1 = SumNode()
    child_2 = SumNode()
    child_3 = SumNode()

    child_1_eval = numpy.random.rand(n_instances)
    child_2_eval = numpy.random.rand(n_instances)
    child_3_eval = numpy.random.rand(n_instances)

    assoc = {child_1: child_1_eval,
             child_2: child_2_eval,
             child_3: child_3_eval}

    prod_node = ProductNode()
    prod_node.add_child(child_1)
    prod_node.add_child(child_2)
    prod_node.add_child(child_3)

    node_eval = evaluate_product_node(prod_node,  assoc, n_instances)
    print(node_eval)
    sum_child_array = child_1_eval + child_2_eval + child_3_eval
    assert_array_almost_equal(node_eval, sum_child_array)


def test_evaluate_sum_node():

    child_1 = ProductNode()
    child_2 = ProductNode()
    child_3 = ProductNode()

    child_1_eval = numpy.random.rand(n_instances)
    child_2_eval = numpy.random.rand(n_instances)
    child_3_eval = numpy.random.rand(n_instances)

    # child_1_eval = numpy.array([1, 1, 1, 1, 1, 1, 1, 1])
    # child_2_eval = numpy.array([1, 1, 1, 1, 1, 1, 1, 1])
    # child_3_eval = numpy.array([1, 1, 1, 1, 1, 1, 1, 1])

    assoc = {child_1: child_1_eval,
             child_2: child_2_eval,
             child_3: child_3_eval}

    weights = numpy.random.rand(len(assoc))

    sum_node = SumNode()
    for i, child in enumerate(assoc):
        sum_node.add_child(child, weights[i])

    assert_array_almost_equal(sum_node.log_weights, numpy.log(weights))
    node_eval = evaluate_sum_node(sum_node,  assoc, n_instances)
    print(node_eval)
    weighted_sum_child_array = numpy.zeros(n_instances)
    for i, child in enumerate(assoc):
        weighted_sum_child_array += weights[i] * numpy.exp(assoc[child])

    assert_array_almost_equal(node_eval, numpy.log(weighted_sum_child_array))


def test_ml_evaluation():
    input_vec = numpy.array([[0., 0., 0.],
                             [0., 0., 0.],
                             [0., 1., 1.],
                             [1., 1., 1.]]).T

    ind_node_1 = CategoricalIndicatorNode(var=0, var_val=0)
    ind_node_2 = CategoricalIndicatorNode(var=0, var_val=1)
    ind_node_3 = CategoricalIndicatorNode(var=1, var_val=0)
    ind_node_4 = CategoricalIndicatorNode(var=1, var_val=1)
    ind_node_5 = CategoricalIndicatorNode(var=2, var_val=0)
    ind_node_6 = CategoricalIndicatorNode(var=2, var_val=1)

    input_layer = CategoricalInputLayer(nodes=[ind_node_1,
                                               ind_node_2,
                                               ind_node_3,
                                               ind_node_4,
                                               ind_node_5,
                                               ind_node_6])

    n_nodes_layer_1 = 6
    layer_1_sum_nodes = [SumNode() for i in range(n_nodes_layer_1)]
    layer_1_sum_nodes[0].add_child(ind_node_1, 0.6)
    layer_1_sum_nodes[0].add_child(ind_node_2, 0.4)
    layer_1_sum_nodes[1].add_child(ind_node_1, 0.3)
    layer_1_sum_nodes[1].add_child(ind_node_2, 0.7)
    layer_1_sum_nodes[2].add_child(ind_node_3, 0.1)
    layer_1_sum_nodes[2].add_child(ind_node_4, 0.9)
    layer_1_sum_nodes[3].add_child(ind_node_3, 0.7)
    layer_1_sum_nodes[3].add_child(ind_node_4, 0.3)
    layer_1_sum_nodes[4].add_child(ind_node_5, 0.5)
    layer_1_sum_nodes[4].add_child(ind_node_6, 0.5)
    layer_1_sum_nodes[5].add_child(ind_node_5, 0.2)
    layer_1_sum_nodes[5].add_child(ind_node_6, 0.8)

    layer_1 = SumLayer(layer_1_sum_nodes)

    n_nodes_layer_2 = 4
    layer_2_prod_nodes = [ProductNode() for i in range(n_nodes_layer_2)]
    layer_2_prod_nodes[0].add_child(layer_1_sum_nodes[0])
    layer_2_prod_nodes[0].add_child(layer_1_sum_nodes[2])
    layer_2_prod_nodes[0].add_child(layer_1_sum_nodes[4])
    layer_2_prod_nodes[1].add_child(layer_1_sum_nodes[1])
    layer_2_prod_nodes[1].add_child(layer_1_sum_nodes[3])
    layer_2_prod_nodes[1].add_child(layer_1_sum_nodes[5])
    layer_2_prod_nodes[2].add_child(layer_1_sum_nodes[0])
    layer_2_prod_nodes[2].add_child(layer_1_sum_nodes[2])
    layer_2_prod_nodes[2].add_child(layer_1_sum_nodes[5])
    layer_2_prod_nodes[3].add_child(layer_1_sum_nodes[1])
    layer_2_prod_nodes[3].add_child(layer_1_sum_nodes[3])
    layer_2_prod_nodes[3].add_child(layer_1_sum_nodes[4])

    layer_2 = ProductLayer(layer_2_prod_nodes)

    root = SumNode()
    root.add_child(layer_2_prod_nodes[0], 0.2)
    root.add_child(layer_2_prod_nodes[1], 0.4)
    root.add_child(layer_2_prod_nodes[2], 0.15)
    root.add_child(layer_2_prod_nodes[3], 0.25)

    layer_3 = SumLayer([root])

    spn = Spn(input_layer=input_layer,
              layers=[layer_1, layer_2, layer_3])

    print(spn)
    res = spn.eval(input_vec)
    print('First evaluation')
    print(res)

    root = spn.root()
    node_evals = ml_evaluation(spn, input_vec.T, nodes_to_eval={root})

    print(node_evals)
    res_vec = numpy.array(res).flatten()
    print(res_vec)
    assert_array_almost_equal(node_evals[root], res_vec)

    #
    # store previous sum nodes weights
    old_weights = [node.weights for node in layer_1_sum_nodes]

    nodes_to_evaluate = set(layer_1_sum_nodes)
    nodes_to_evaluate.add(root)

    node_evals = ml_evaluation(spn, input_vec.T,
                               nodes_to_eval=None,
                               # nodes_to_skip_updating=nodes_to_skip,
                               update_weights=True)
    # assert_array_almost_equal(node_evals[root], res_vec)
    print(node_evals)
    assert node_evals == {}

    print(node_evals)
    print(spn)

    #
    # evaluating and changing only the root
    node_evals = ml_evaluation(spn, input_vec.T,
                               nodes_to_eval={root},
                               update_weights=True)
    print(node_evals)
    print(spn)
    new_weights = [node.weights for node in layer_1_sum_nodes]
    for old_w, new_w in zip(old_weights, new_weights):
        assert_array_almost_equal(old_w, new_w)

    node_evals = ml_evaluation(spn, input_vec.T,
                               nodes_to_eval=nodes_to_evaluate,
                               update_weights=True)
    print(node_evals)
    print(spn)
    assert_array_almost_equal(root.weights,
                              numpy.array([0.25 for i in range(len(layer_2_prod_nodes))]))
