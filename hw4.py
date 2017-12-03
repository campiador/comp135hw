# hw4.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This code implements a neural network and learns it based on (w, d) hidden nodes and (train, test) data
#
#

from __future__ import division

import argparse

import math


from models.neural_network import NeuralNetwork
from models.neural_node import NeuralNode

N_ITER = 3000



parser = argparse.ArgumentParser()
parser.parse_args()





def construct_network_and_initialize_weights(width, depth, input_layer, output_layer):
    return NeuralNetwork(width, depth, input_layer, output_layer)


def extract_input_and_output_layer(train_data):
    #TODO parse data to get input and output layers
    return [NeuralNode(0, 0)], [NeuralNode(0, 0)]


def learn(width, depth, train_data, test_data):

    (input_layer, output_layer) = extract_input_and_output_layer(train_data)

    network = construct_network_and_initialize_weights(width, depth, input_layer, output_layer)

    for i in range(0, N_ITER):
        for example in train_data:
            network.update_weights_using_forward_and_backpropagation(example)

        network.calculate_training_error_rate()








def sigmoid(x):
    x = max(-50, x)  # to avoid numerical issues
    return 1/(1 + math.exp(x))


if __name__ == '__main__':
    args = parser.parse_args()

# TODO: Parse these
    w = 2
    d = 1

    train_data = []

    test_data = []

    learn(w, d, train_data, test_data)

