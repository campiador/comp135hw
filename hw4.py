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
from models.example import Example
from models.neural_network import NeuralNetwork
from models.neuron import Neuron
from parser.arffparser import parse_file_and_extract_examples_and_number_of_classes_and_features, \
    parse_file_and_extract_examples, extract_output_classes

N_ITER = 2
INPUT_FILES_DIR = "./input/hw4"

FILE_838 = "838.arff"
FILE_OPT_DIGITS_TRAIN = "optdigits_train.arff"
FILE_OPT_DIGITS_TEST = "optdigits_test.arff"


def construct_network_and_initialize_weights(width, depth, input_layer, output_layer):
    return NeuralNetwork(width, depth, input_layer, output_layer)


def calculate_input_and_output_layers(train_data_examples, n_input_nodes, n_output_nodes, output_classes):
    input_layer = [Neuron() for _ in range(0, n_input_nodes)]
    # FIXME: should I write output values to the input unit? probably not
    # for i, neuron in enumerate(input_layer):
    #     neuron.output = train_data_examples[i]

    output_layer = [Neuron() for _ in range(0, n_output_nodes)]

    return input_layer, output_layer


def calculate_error_rate(number_of_training_errors_per_example, number_of_examples):
    return number_of_training_errors_per_example / number_of_examples


def learn(width, depth, train_data_examples, test_data_examples, n_input_nodes, n_output_nodes, output_classes):

    (input_layer, output_layer) = \
        calculate_input_and_output_layers(train_data_examples, n_input_nodes, n_output_nodes, output_classes)

    network = construct_network_and_initialize_weights(width, depth, input_layer, output_layer)

    number_of_training_examples = len(train_data_examples)
    training_error_rates = []

    number_of_test_examples = len(test_data_examples)
    test_error_rates = []

    for i in range(0, N_ITER):
        print "ITERATION:", i
        number_of_training_mistakes = 0
        for train_example in train_data_examples:
            # print "example:", train_example
            network.init_desired_onehot_labels_for_output_nodes(train_example.label, output_classes)
            number_of_training_mistakes += \
                network.update_weights_using_forward_and_backpropagation_return_1_if_mistake(train_example)
        print "mistake#", number_of_training_mistakes
        training_error_rate = calculate_error_rate(number_of_training_mistakes, number_of_training_examples)
        training_error_rates.append(training_error_rate)

        number_of_test_errors = 0

        for test_example in test_data_examples:
            # Note: assuming the same output classes for test and training
            network.init_desired_onehot_labels_for_output_nodes(test_example.label, output_classes)
            network.forward_feed_input_and_calculate_node_output_values(test_example)
            if network.was_there_a_mistake_in_output():
                number_of_test_errors += 1
        if len(test_data_examples) != 0:
            test_error_rate = calculate_error_rate(number_of_test_errors, number_of_test_examples)
            test_error_rates.append(test_error_rate)


    print "training and test error rates for all iterations:", training_error_rates, test_error_rates
    print len(training_error_rates), len(test_error_rates)
        #TODO: what to do with above variable?

RUN_MY_EX = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    args = parser.parse_args()

# calculate these from input?
    w = 5
    d = 1

    # (examples, n_classes, n_features, output_classes) = \
    #     parse_file_and_extract_examples_and_number_of_classes_and_features(INPUT_FILES_DIR, FILE_838)

    # (opt_train_examples, _, _, _) = \
    (examples, n_classes, n_features, output_classes) = \
        parse_file_and_extract_examples_and_number_of_classes_and_features(INPUT_FILES_DIR, FILE_OPT_DIGITS_TRAIN)

    (opt_test_examples, _, _, _) = \
        parse_file_and_extract_examples_and_number_of_classes_and_features(INPUT_FILES_DIR, FILE_OPT_DIGITS_TEST)


    if LOG_VERBOSE:
        print "examples after parsing file:", examples
        for example in examples:
            print example.features, example.label

    train_data_examples = examples


    test_data = [Example(1000, [0, 1, 0, 0, 0, 0, 0, 0], "2")]


    if RUN_MY_EX:
        my_example_train_data = [Example(0, [2, 3], "\n0")]
        my_ex_num_features = 2
        my_ex_n_classes = 1
        my_ex_output_classes = ["0"]
        learn(w, d, my_example_train_data, test_data, my_ex_num_features, my_ex_n_classes, my_ex_output_classes)
    else:
        learn(w, d, train_data_examples, opt_test_examples, n_features, n_classes, output_classes)

