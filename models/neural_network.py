# hw4.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This module models a neural network
#
import random

import numpy

from log.log import LOG_VERBOSE, LOG_DEVELOPER

numpy.random.seed(0)


from models.neuron import Neuron


class NeuralNetwork():
    def __init__(self, width, depth, input_layer, output_layer):
        self.weights = []
        self.width = width
        self.depth = depth

        self.node_layers = []

        self.init_node_layers(self.depth, self.width, input_layer, output_layer)
        if LOG_DEVELOPER:
            for i, layer in enumerate(self.node_layers):
                print "layer {}".format(i)
                print layer

        self.initialize_weights_randomly(self.width, self.depth, len(input_layer), len(output_layer))

        if LOG_DEVELOPER:
            print "\nweights"
            print self.weights


    def update_weights_using_forward_and_backpropagation_return_train_errors(self, example):
        self.forward_propagate_input_and_calculate_node_output_values(example)
        self.backward_propagate_and_calculate_deltas()
        self.forward_update_weights()

        # TODO: calculate training errors
        return 0

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
            this_layer = self.node_layers[weight_layer_index]
            next_layer = self.node_layers[weight_layer_index + 1]

            self.weights.append([random.uniform(-0.1, 0.1) for _ in range(0, len(this_layer) * len(next_layer))])

    def init_node_layers(self, depth, width, input_layer, output_layer):
        # First layer is input layer
        self.node_layers.append(input_layer)

        # Now hidden layers
        for i in range(0, depth):
            layer = []
            for j in range(0, width):
                layer.append(Neuron())
            self.node_layers.append(layer)

        # Last layer is output layer
        self.node_layers.append(output_layer)










