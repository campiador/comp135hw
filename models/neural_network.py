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

random.seed(162)

from models.neuron import Neuron

DEFAULT_LEARNING_RATE = 0.1



class NeuralNetwork():
    """ data representation:
    self.weights is a an array of d + 1 inner arrays.
    each inner array sits between two layers
    """
    def __init__(self, width, depth, input_layer, output_layer):

        self.weights = []
        self.width = width
        self.depth = depth

        self.node_layers = []

        self.init_node_layers(self.depth, self.width, input_layer, output_layer)

        self.initialize_weights_randomly(self.width, self.depth, len(input_layer), len(output_layer))

        self.learning_rate = DEFAULT_LEARNING_RATE

        if LOG_DEVELOPER:
            print "initialized network!"
        if LOG_VERBOSE:
            print self

    def __str__(self):
        str = ""
        for i, layer in enumerate(self.node_layers):
            str += "\nlayer {}".format(i)
            str += "{}".format(layer)
            if i < len(self.weights):
                pretty_weights = ["%.8f" % item for item in self.weights[i]]
                str += "\nweights:{}".format(pretty_weights)
        return str

    def __repr__(self):
        str = ""
        for i, layer in enumerate(self.node_layers):
            str += "\nlayer {}".format(i)
            str += "{}".format(layer)
            if i < len(self.weights):
                pretty_weights = ["%.8" % item for item in self.weights[i]]
                str += "weights:{}".format(pretty_weights)
        return str

    def update_weights_using_forward_and_backpropagation_return_1_if_mistake(self, example):

        self.forward_feed_input_and_calculate_node_output_values(example)
        # if LOG_DEVELOPER:
        #     print "*** after feeding forward: ***"
        #     print self

        self.backward_propagate_and_calculate_deltas()
        # if LOG_DEVELOPER:
        #     print "after backpropagate:"
        #     print self

        self.forward_update_weights()
        # if LOG_DEVELOPER:
        #     print "after weight update:"
        #     print self

        # Note: we already set the onehotlabels of data before calling the function we are in

        is_mistake = self.is_there_a_mistake_in_output_prediction()

        if is_mistake:
            return 1
        else:  # correct prediction!
            return 0

    def forward_feed_input_and_calculate_node_output_values(self, example):
        """ calculate s for each node, and then calculate x for it """

        # for input_node in input_layer:
        for i, input_node in enumerate(self.node_layers[0]):
            input_node.output = example.features[i]

        # for layers after the input, calculate the sums
        for index_current_layer, noninput_layer in enumerate(self.node_layers):
            if index_current_layer == 0:
                continue  # already addressed input layer in the first for loop
        # Hidden layers and output layer

            for node_index_in_current_layer, node_in_current_layer in enumerate(noninput_layer):
                lower_layer = self.get_lower_layer(index_current_layer)
                node_in_current_layer.sum_of_node_inputs = 0  # This is all logic code was changed from version 1 to 2
                for lower_layer_node_index, lower_layer_node in enumerate(lower_layer):
                    node_in_current_layer.sum_of_node_inputs += \
                        self.get_weight(index_current_layer - 1, lower_layer_node_index, node_index_in_current_layer) \
                        * lower_layer_node.output
                node_in_current_layer.output = sigmoid(node_in_current_layer.sum_of_node_inputs)
                node_in_current_layer.dp = node_in_current_layer.output * (1 - node_in_current_layer.output)

    def initialize_weights_randomly(self, width, depth, input_layer_len, output_layer_len):

        for weight_layer_index in range(0, depth + 1):
            this_layer = self.node_layers[weight_layer_index]
            next_layer = self.node_layers[weight_layer_index + 1]

            self.weights.append([random.uniform(-0.1, 0.1) for _ in range(0, len(this_layer) * len(next_layer))])
            # if LOG_VERBOSE:
            #     print len(self.weights[-1])

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

    def get_lower_layer(self, l):
        return self.node_layers[l-1]

    def get_weight(self, relatively_lower_layer_index, relatively_lower_layer_node_index,
                   relatively_higher_layer_node_index):

        # if LOG_VERBOSE:
        #     print "lower layer index:", relatively_lower_layer_index
        #     print "lower layer node index:", relatively_lower_layer_node_index
        #     print "relatively higher level index", relatively_higher_layer_node_index
        n = len(self.node_layers[relatively_lower_layer_index])
        return self.weights[relatively_lower_layer_index][n * relatively_higher_layer_node_index + relatively_lower_layer_node_index]


    #
    # def get_weight_backward(self, lower_layer_index, i, current_node_index):
    #     return self.weights[lower_layer_index][(i * m) + j]

    def backward_propagate_and_calculate_deltas(self):

        # calculate delta for last layer
        for index, last_layer_node in enumerate(self.node_layers[-1]):
            last_layer_node.delta = -(last_layer_node.dp) * (last_layer_node.onehot_label - last_layer_node.output)

        # calculate delta for hidden layers. top down
        for reversed_index, current_layer in enumerate(reversed(self.node_layers[1:-1])):

            higher_layer_index = len(self.node_layers) - 1 - reversed_index

            current_layer_index = higher_layer_index - 1
            # if LOG_VERBOSE:
            #     print "bpp for hidden layer:", current_layer_index

            higher_layer = self.node_layers[higher_layer_index]

            for current_layer_node_index, current_layer_node in enumerate(current_layer):
                higher_layer_delta_times_w_sum = 0

                for higher_layer_node_index, higher_layer_node in enumerate(higher_layer):
                    higher_layer_node_delta = higher_layer_node.delta
                    higher_layer_weight = self.get_weight(current_layer_index, current_layer_node_index,
                                                          higher_layer_node_index)

                    higher_layer_delta_times_w_sum += higher_layer_node_delta * higher_layer_weight

                new_delta = current_layer_node.dp * higher_layer_delta_times_w_sum

                self.node_layers[current_layer_index][current_layer_node_index].delta = new_delta

    def init_desired_onehot_labels_for_output_nodes(self, label, output_classes):
        for position_of_neuron_in_layer, output_node in enumerate(self.node_layers[-1]):  # output layer
            output_node.set_onehot_label(position_of_neuron_in_layer, output_classes, label)

    def forward_update_weights(self):

        for index_weight_layer, weight_layer in enumerate(self.weights):

            for weight_index_in_layer, _ in enumerate(weight_layer):

                from_node, to_node = self.get_from_and_to_nodes_by_weight(index_weight_layer, weight_index_in_layer)

                self.weights[index_weight_layer][weight_index_in_layer] \
                    -= (self.learning_rate * from_node.output * to_node.delta)

    def get_from_and_to_nodes_by_weight(self, index_weight_layer, weight_index_in_layer):

        len_lower_node_layer = len(self.node_layers[index_weight_layer])

        index_node_in_higher_layer = weight_index_in_layer / len_lower_node_layer

        index_node_in_lower_layer = weight_index_in_layer % len_lower_node_layer

        from_node = self.node_layers[index_weight_layer][index_node_in_lower_layer]
        to_node = self.node_layers[index_weight_layer + 1][index_node_in_higher_layer]

        return from_node, to_node

    def is_there_a_mistake_in_output_prediction(self):
        output_layer = self.node_layers[-1]

        output_values = [node.output for node in output_layer]
        predicted_index = output_values.index(max(output_values))

        output_labels = [node.onehot_label for node in output_layer]

        actual_index = -1

        for i, onehot_label in enumerate(output_labels):
            if onehot_label == 1:
                actual_index = i
                break

        if actual_index == -1:
            raise AssertionError, "No datapoint has been assigned onehot value 1. This suggests a bug in onehot code."
            exit(1)

        if predicted_index == actual_index:
            return False
        else:
            return True





