from collections import deque
from collections import defaultdict

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalIndicatorNode

from spn.linked.layers import CategoricalIndicatorLayer
from spn.linked.layers import SumLayer
from spn.linked.layers import ProductLayer

from spn.linked.spn import Spn as LinkedSpn

import numpy

import itertools


def topological_layer_sort(layers):
    """
    layers is a sequence of layers
    """

    #
    #
    layers_dict = {layer: layer.input_layers for layer in layers}

    sorted_layers = []

    while layers_dict:

        acyclic = False
        temp_layers_dict = dict(layers_dict)
        for layer, descendants in temp_layers_dict.items():
            for desc_layer in descendants:
                if desc_layer in layers_dict:
                    break
            else:
                acyclic = True
                del layers_dict[layer]
                sorted_layers.append(layer)

        if not acyclic:
            raise RuntimeError("A cyclic dependency occurred")

    return sorted_layers
