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
from models.neuron import Neuron

N_ITER = 3000



parser = argparse.ArgumentParser()
parser.parse_args()





def construct_network_and_initialize_weights(width, depth, input_layer, output_layer):
    return NeuralNetwork(width, depth, input_layer, output_layer)


def extract_input_and_output_layer(train_data):
    #TODO parse data to get input and output layers
    return [Neuron()], [Neuron()]


def learn(width, depth, train_data, test_data):

    (input_layer, output_layer) = extract_input_and_output_layer(train_data)

    network = construct_network_and_initialize_weights(width, depth, input_layer, output_layer)

    number_of_examples = len(train_data)
    training_error_rates = []

    for i in range(0, N_ITER):
        number_of_training_errors = 0
        for example in train_data:
            number_of_training_errors += \
                network.update_weights_using_forward_and_backpropagation_return_train_errors(example)

        training_error_rate = network.calculate_training_error_rate(number_of_training_errors, number_of_examples)
        training_error_rates.append(training_error_rate)
        #TODO: what to do with above variable?


def sigmoid(x):
    x = max(-50, x)  # to avoid numerical issues
    return 1/(1 + math.exp(x))


if __name__ == '__main__':
    args = parser.parse_args()

# TODO: Parse these
    w = 2
    d = 5

    train_data = []

    test_data = []

    learn(w, d, train_data, test_data)

