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

import sys

from graphics.plot import plot_x_y_line
from graphics.subplotable import SubPlotable
from log.log import LOG_VERBOSE, LOG_DEVELOPER, LOG_CLIENT
from models.example import Example
from models.neural_network import NeuralNetwork
from models.neuron import Neuron
from parser.arffparser import parse_file_and_extract_examples_and_number_of_classes_and_features, \
    parse_file_and_extract_examples, extract_output_classes

N_ITER_DEFAULT = 1
INPUT_FILES_DIR = "./input/hw4"

FILE_838 = "838.arff"
FILE_OPT_DIGITS_TRAIN = "optdigits_train.arff"
FILE_OPT_DIGITS_TEST = "optdigits_test.arff"


def construct_network_and_initialize_weights(width, depth, input_layer, output_layer):
    return NeuralNetwork(width, depth, input_layer, output_layer)


def create_input_and_output_layers(n_input_nodes, n_output_nodes):
    input_layer = [Neuron() for _ in range(0, n_input_nodes)]
    output_layer = [Neuron() for _ in range(0, n_output_nodes)]

    return input_layer, output_layer


def calculate_error_rate(number_of_training_errors_per_example, number_of_examples):
    return number_of_training_errors_per_example / number_of_examples


def learn_and_return_test_train_errors(width, depth, train_data_examples, test_data_examples, n_input_nodes,
                                       n_output_nodes, output_classes, n_iterations):

    if LOG_CLIENT:
        print "\ncreating a neural network for w:{}, d:{}, and running it for n:{} iterations"\
            .format(width, depth, n_iterations)


    (input_layer, output_layer) = create_input_and_output_layers(n_input_nodes, n_output_nodes)

    network = construct_network_and_initialize_weights(width, depth, input_layer, output_layer)

    number_of_training_examples = len(train_data_examples)
    training_error_rates = []

    number_of_test_examples = len(test_data_examples)
    test_error_rates = []

    for i in range(0, n_iterations):
        print "ITERATION:", i
        number_of_training_mistakes = 0
        for train_example in train_data_examples:
            # print "example:", train_example
            network.init_desired_onehot_labels_for_output_nodes(train_example.label, output_classes)
            number_of_training_mistakes += \
                network.update_weights_using_forward_and_backpropagation_return_1_if_mistake(train_example)
        print "mistakes :", number_of_training_mistakes
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

    if LOG_CLIENT:
        print "training and test error rates for all iterations:", training_error_rates, test_error_rates
        print "training error rate after {} iterations: {}".format(n_iterations, training_error_rates[-1])
    if len(test_error_rates) > 0:
        print "test error rate after {} iterations: {}".format(n_iterations, test_error_rates[-1])
    return training_error_rates, test_error_rates


RUN_EXAMPLE_FROM_CLASS = False


def run_program(train_input_file, test_input_file, w_list, d_list, iteration_count):

    (train_examples, n_classes, n_features, output_classes) = \
        parse_file_and_extract_examples_and_number_of_classes_and_features(INPUT_FILES_DIR, train_input_file)

    if LOG_CLIENT:
        print "\nfile:{}".format(train_input_file)

    if LOG_VERBOSE:
        print "examples after parsing file:", train_input_file
        for example in train_examples:
            print example.features, example.label

    if test_input_file == None:
        test_input_examples = []
    else:
        (test_input_examples, _, _, _) = \
            parse_file_and_extract_examples_and_number_of_classes_and_features(INPUT_FILES_DIR, test_input_file)

    w_results = []
    for w in w_list:
        d_results = []
        for d in d_list:
            train_test_errors = learn_and_return_test_train_errors(w, d, train_examples, test_input_examples,
                                                                             n_features, n_classes, output_classes,
                                                                             iteration_count)
            d_results.append(train_test_errors)
        w_results.append(d_results)

    return w_results


def plot_part1(w_d_results_part1):
    # total = len(w_d_results_part1)

    x_values = []
    train_values = []
    test_values = []

    # 3 indexes: w, d, train/test
    results = w_d_results_part1[0][0][0]

    print "results:", results

    for i, result in enumerate(results):
        x_values.append(i)
        train_values.append(result)
        # test_values.append(results[1])


    subplotable_train = SubPlotable("trainerror ", x_values, train_values, [0 for _ in train_values])
    # subplotable_test = SubPlotable("test error", x_values, test_values, [0 for _ in test_values])



    subplotables = [subplotable_train]



    plot_x_y_line("Part 1", "Iteration", "Train Error", subplotables, "part1")



def plot_part2(w_d_results_part2):
    # total = len(w_d_results_part1)

    # x_values = []
    # train_values = []
    # test_values = []

    # 3 indexes: w, d, train/test
    # results = w_d_results_part1[0][0][0]

    print "results:", results
    subplotables = []

    for w_index, results_by_w in w_d_results_part2:
        w = part2_exp1_w[w_index]
        for d_index, results_by_d in results_by_w:
            d = part2_exp1_d[d_index]
            subplotables = []
            for i, result in enumerate(results_by_d):
                total_x_train = len(result[0])

                x_values = [x for x in range(1, total_x_train + 1)]
                train_values = result[0]
                test_values = result[1]

                subplotable_train = SubPlotable("train error ", x_values, train_values, [0 for _ in train_values])
                subplotable_test = SubPlotable("test error", x_values, test_values, [0 for _ in test_values])
                subplotables.append(subplotable_train)
                subplotables.append(subplotable_test)
                plot_x_y_line("Part 2-1-1", "Iteration", "Train and Test Error for w:{} and d:{}",
                              subplotables, "part2_1_1_w{}_d{}".format(w, d))


# CONSTANTS

# PART 1
part1_input_file = FILE_838
part1_w = [3]
part1_d = [1]
part1_iteration = 3000
# PART 2
part2_input_file = FILE_OPT_DIGITS_TRAIN
part2_test_file = FILE_OPT_DIGITS_TEST

part2_exp1_w = [5, 10]
# part2_exp1_w = [5, 10, 15, 20, 30, 40]
part2_exp1_d = [3]

part2_exp2_w = [10]
# part2_exp2_d = [0, 1, 2, 3, 4, 5]
part2_exp2_d = [0, 1]

part2_iteration = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    args = parser.parse_args()

    # For manula use with w, d, traindata, testdata args
    if len(sys.argv) > 1:
        w = int(sys.argv[1])
        d = int(sys.argv[2])
        train_data_file = sys.argv[3]
        test_data_file = sys.argv[4]

        run_program(train_data_file, test_data_file, [w], [d], N_ITER_DEFAULT)

    w_d_results_part1 = run_program(part1_input_file, None, part1_w, part1_d, part1_iteration)
    plot_part1(w_d_results_part1)

    w_d_results_part2_exp1 = run_program(part2_input_file, part2_test_file, part2_exp1_w, part2_exp1_d, part2_iteration)
    w_d_results_part2_exp2 = run_program(part2_input_file, part2_test_file, part2_exp2_w, part2_exp2_d, part2_iteration)
    #

    # plot_part2_1()
    # plot_part2_2()




   # if RUN_MY_EX:
    #     my_example_train_data = [Example(0, [2, 3], "\n0")]
    #     my_ex_num_features = 2
    #     my_ex_n_classes = 1
    #     my_ex_output_classes = ["0"]
    #     learn(w, d, my_example_train_data, test_examples, my_ex_num_features, my_ex_n_classes, my_ex_output_classes)


