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

from log.log import LOG_VERBOSE, LOG_DEVELOPER
from models.neural_network import NeuralNetwork
from models.neuron import Neuron
from parser.arffparser import parse_file_and_extract_examples_and_number_of_classes_and_features, \
    parse_file_and_extract_examples

N_ITER = 3000
INPUT_FILES_DIR = "./input/hw4"

FILE_838 = "838.arff"
FILE_OPT_DIGITS_TRAIN = "optdigits_train.arff"
FILE_OPT_DIGITS_TEST = "optdigits_test.arff"


def construct_network_and_initialize_weights(width, depth, input_layer, output_layer):
    return NeuralNetwork(width, depth, input_layer, output_layer)


def calculate_input_and_output_layers(train_data_examples, n_input_nodes, n_output_nodes):
    input_layer = [Neuron() for _ in range(0, n_input_nodes)]
    # FIXME: should I write output values to the input unit? probably not
    # for i, neuron in enumerate(input_layer):
    #     neuron.output = train_data_examples[i]

    output_layer = [Neuron() for _ in range(0, n_output_nodes)]

    return input_layer, output_layer


def learn(width, depth, train_data_examples, test_data, n_input_nodes, n_output_nodes):

    (input_layer, output_layer) = calculate_input_and_output_layers(train_data_examples, n_input_nodes, n_output_nodes)

    network = construct_network_and_initialize_weights(width, depth, input_layer, output_layer)

    number_of_examples = len(train_data_examples)
    training_error_rates = []

    for i in range(0, N_ITER):
        number_of_training_errors = 0
        for example in train_data_examples:
            number_of_training_errors += \
                network.update_weights_using_forward_and_backpropagation_return_train_errors(example)
        exit(2)
        training_error_rate = network.calculate_training_error_rate(number_of_training_errors, number_of_examples)
        training_error_rates.append(training_error_rate)
        #TODO: what to do with above variable?


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    args = parser.parse_args()

# calculate these from input?
    w = 3
    d = 1

    (examples, n_classes, n_features) = parse_file_and_extract_examples_and_number_of_classes_and_features(INPUT_FILES_DIR, FILE_838)

    if LOG_VERBOSE:
        print "examples after parsing file:", examples
        for example in examples:
            print example.features, example.label

    train_data = examples


    test_data = []

    learn(w, d, train_data, test_data, n_features, n_classes)
