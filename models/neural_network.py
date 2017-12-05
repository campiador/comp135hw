# neural_network.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This module models a neural network
#
import random

import numpy

from log.log import LOG_VERBOSE, LOG_DEVELOPER, LOG_CLIENT
from statistics_numeric_methods.statistics import sigmoid

numpy.random.seed(0)

from models.neuron import Neuron

DEFAULT_LEARNING_RATE = 0.1

class NeuralNetwork():
    def __init__(self, width, depth, input_layer, output_layer):

        self.weights = []
        self.width = width
        self.depth = depth

        self.node_layers = []

        self.init_node_layers(self.depth, self.width, input_layer, output_layer)

        self.initialize_weights_randomly(self.width, self.depth, len(input_layer), len(output_layer))

        self.learning_rate = DEFAULT_LEARNING_RATE

        print self

    def __str__(self):
        str = ""
        for i, layer in enumerate(self.node_layers):
            str += "\nlayer {}".format(i)
            str += "{}".format(layer)
            if i < len(self.weights):
                pretty_weights = ["%.2f" % item for item in self.weights[i]]
                str += "\nweights:{}".format(pretty_weights)
        return str

    def __repr__(self):
        str = ""
        for i, layer in enumerate(self.node_layers):
            str += "\nlayer {}".format(i)
            str += "{}".format(layer)
            if i < len(self.weights):
                pretty_weights = ["%.2f" % item for item in self.weights[i]]
                str += "weights:{}".format(pretty_weights)
        return str

    def update_weights_using_forward_and_backpropagation_return_train_errors(self, example):

        self.forward_feed_input_and_calculate_node_output_values(example)

        print self

        self.backward_propagate_and_calculate_deltas()

        print self

        self.forward_update_weights()

        # TODO: calculate training errors
        return 0

    def forward_feed_input_and_calculate_node_output_values(self, example):
        """ calculate s for each node, and then calculate x for it """
        # for input_node in input_layer:
        for i, input_node in enumerate(self.node_layers[0]):
            input_node.output = example.features[i]

        # for layers after the input, calculate the sums
        for index_current_layer, noneinput_layer in enumerate(self.node_layers):
            if index_current_layer == 0:
                continue  # already addressed input layer in the first for loop
            for node_index_in_current_layer, node_in_current_layer in enumerate(noneinput_layer):
                lower_layer = self.get_lower_layer(index_current_layer)
                for lower_layer_node_index, lower_layer_node in enumerate(lower_layer):
                    node_in_current_layer.sum_of_node_inputs += \
                        self.get_weight(index_current_layer - 1, lower_layer_node_index, node_index_in_current_layer) \
                        * lower_layer_node.output
                node_in_current_layer.output = sigmoid(node_in_current_layer.sum_of_node_inputs)

    def initialize_weights_randomly(self, width, depth, input_layer_len, output_layer_len):

        for weight_layer_index in range(0, depth + 1):
            this_layer = self.node_layers[weight_layer_index]
            next_layer = self.node_layers[weight_layer_index + 1]

            self.weights.append([random.uniform(-0.1, 0.1) for _ in range(0, len(this_layer) * len(next_layer))])

    def init_node_layers(self, depth, width, input_layer, output_layer):
        # First layer is input layer
        #FIXME: was first layer added to node_layers twice? do we need to check something
        self.node_layers.append(input_layer)

        # Now hidden layers
        for i in range(0, depth):
            layer = []
            for j in range(0, width):
                layer.append(Neuron())
            self.node_layers.append(layer)

        # Last layer is output layer
        self.node_layers.append(output_layer)

    def calculate_training_error_rate(self, number_of_training_errors, number_of_examples):
        raise NotImplementedError, "not implemented"

    def get_lower_layer(self, l):
        return self.node_layers[l-1]

    def get_weight(self, lower_layer_index, lower_layer_node_index, current_layer_node_index):
        n = len(self.node_layers[lower_layer_index])
        return self.weights[lower_layer_index][n * current_layer_node_index + lower_layer_node_index]


    #
    # def get_weight_backward(self, lower_layer_index, i, current_node_index):
    #     return self.weights[lower_layer_index][(i * m) + j]

    def backward_propagate_and_calculate_deltas(self):

        # calculate delta for last layer
        for index, last_layer_node in enumerate(self.node_layers[-1]):
            last_layer_node.delta = -(last_layer_node.onehot_label - last_layer_node.output) * \
                                    last_layer_node.output * (1 - last_layer_node.output)

        # calculate delta for hidden layers. top down
        for reversed_index, current_layer in enumerate(reversed(self.node_layers[1:-1])):

            higher_layer_index = len(self.node_layers) - 1 - reversed_index

            current_layer_index = higher_layer_index - 1

            higher_layer = self.node_layers[higher_layer_index]

            for current_layer_node_index, current_layer_node in enumerate(current_layer):
                new_delta = 0

                higher_layer_delta_times_w_sum = 0

                for higher_layer_node_index, higher_layer_node in enumerate(higher_layer):
                    higher_layer_node_delta = higher_layer_node.delta
                    higher_layer_weight = self.get_weight(current_layer_index, current_layer_node_index,
                                                          higher_layer_node_index)
                    higher_layer_delta_times_w_sum += higher_layer_node_delta * higher_layer_weight

                new_delta = current_layer_node.output * (1- current_layer_node.output) * higher_layer_delta_times_w_sum

                self.node_layers[current_layer_index][current_layer_node_index].d = new_delta


    def init_onehot_labels_for_output_nodes(self, label, output_classes):
        for position_of_neuron_in_layer, node in enumerate(self.node_layers[-1]):
            node.set_onehot_label(position_of_neuron_in_layer, output_classes, label)

    # CONTINUE HERE
    def forward_update_weights(self):
        for index_current_layer, noneinput_layer in enumerate(self.node_layers):
            if index_current_layer == 0:
                continue  # already addressed input layer in the first for loop
            for node_index_in_current_layer, node_in_current_layer in enumerate(noneinput_layer):
                lower_layer = self.get_lower_layer(index_current_layer)
                for lower_layer_node_index, lower_layer_node in enumerate(lower_layer):
                    node_in_current_layer.selfnode= \
                        self.get_weight(index_current_layer - 1, lower_layer_node_index, node_index_in_current_layer) \
                        * self.learning_rate



