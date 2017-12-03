# hw4.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This module models a neural network
#
from numpy.random import random

from models.neural_node import NeuralNode


class NeuralNetwork():
    def __init__(self, width, depth, input_layer, output_layer):
        self.weights = []
        self.width = width
        self.depth = depth

        self.layers = []

        self.init_node_layers(self.depth, self.width, input_layer, output_layer)

        self.initialize_weights_randomly(self.width, self.depth, len(input_layer), len(output_layer))

    def update_weights_using_forward_and_backpropagation(self, example):
        self.forward_propagate_input_and_calculate_node_output_values(example)
        self.backward_propagate_and_calculate_deltas()
        self.forward_update_weights()

    def forward_propagate_input_and_calculate_node_output_values(self, example):
        """ calculate s for each node, and then calculate x for it """
        # for input_node in input_layer:
        #   no work needed here

        for l, hidden_layer in enumerate(self.hidden_layers):
            for i, hidden_node in enumerate(hidden_layer):
                lower_layer = hidden_node.get_lower_layer()
                for lower_layer_node in lower_layer:
                    hidden_node.sum_of_node_inputs += \
                        self.get_weight(lower_layer, hidden_node, lower_layer_node) * lower_layer_node.getoutputvalue()

    def initialize_weights_randomly(self, width, depth, input_layer_len, output_layer_len):
        for weight_layer_index in range(0, depth + 1):
            pass

    def init_node_layers(self, depth, width, input_layer, output_layer):
        self.layers.append(input_layer)

        # Now hidden layers
        for i in range(0, depth):
            layer = []
            for j in range(0, width):
                layer.append(NeuralNode(0, 0))  # random.randrange(-1, 1)

            self.layers.append(layer)

        self.layers.append(output_layer)










