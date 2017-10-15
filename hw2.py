# hw2.py
# by Behnam Heydarshahi, October 2017
# Empirical/Programming Assignment 2
# COMP 135 Machine Learning
#
# This code implements Naive Bayes algorithm with smoothing
# We then evaluate it using cross validation on the task of sentiment analysis
#
# NOTE: To be able to use this codei
#
from __future__ import division

import random

import time

import constants
import plot
import subplotable
import statistics
import vocabulary
from vocabulary import Vocabulary


import re
from numpy import log2, os
import output_class

# CONSTANTS

K = 10
DATASET_FILE_YELP = "yelp_labelled.txt"
DATASET_FILE_IMDB = "imdb_labelled.txt"
DATASET_FILE_AMAZON = "amazon_cells_labelled.txt"

DATASET_FILES = [DATASET_FILE_YELP, DATASET_FILE_IMDB, DATASET_FILE_AMAZON]

SMOOTHING_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# for improving code readability
TRAIN = 0
TEST = 1

# CONTROL VARIABLES

SHORTEN_EXPERIMENT = 0
LIMIT_LINES_TO_10 = 0
ON_SERVER = 1

SANITIZE_LINES = 1
SANITIZE_TOKENS = 1

DEFAULT_SMOOTHING_FACTOR = 1

INPUT_FILES_DIR = "."
def set_environment():
    global INPUT_FILES_DIR
    # Don't put a "/" in the end of the path
    if ON_SERVER:
        INPUT_FILES_DIR = "."
    else:  # On local machine
        INPUT_FILES_DIR = "./sentiment labelled sentences"
    if not os.path.exists("./output/"):
        os.makedirs("./output/")


# Credit: https://stackoverflow.com/questions/10017147/removing-a-list-of-characters-in-string
def sanitize_line_remove_punctuation(line):
    chars_to_remove = [".", ",", "!", "?"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    return re.sub(rx, '', line)


def sanitize_token_to_lowercase(token):
    return token.lower()


def probability_of_word_given_class(word, class_value, vocab, smoothing_factor):
    n_w_in_c = vocab.get_word_count_given_class(word, class_value)
    if n_w_in_c == vocabulary.SKIP_IT:
        return vocabulary.SKIP_IT
    if n_w_in_c == 0 and constants.DEBUG_VERBOSE:
        print "zero count for {} in class {}".format(word, class_value)
    n_c = vocab.get_total_count_for_class(class_value)
    v = vocab.get_vocabulary_size()
    m = smoothing_factor
    if constants.DEBUG_VERBOSE:
        print "(n_w_in_c + m)/(n_c + m*v) = ({} + {})/({} + {}*{}) = {}"\
            .format(n_w_in_c, m, n_c, m, v, (n_w_in_c + m)/(n_c + m*v))
    return (n_w_in_c + m)/(n_c + m*v)


def calculate_score_of_testline_given_class(word_tokens, class_value, vocab, smoothing_factor):
    score = 0
    for word_token in word_tokens:
        if constants.DEBUG_VERBOSE:
            print "getting score for {}, class {}".format(word_token, class_value)
        p = probability_of_word_given_class(word_token, class_value, vocab, smoothing_factor)
        if p == vocabulary.SKIP_IT:
            if constants.DEBUG_VERBOSE:
                print "Skipping {} in class {}".format(word_token, class_value)
            continue
        if constants.DEBUG_VERBOSE and p == 0:
            print "Logging Zero score for {}".format(word_token, class_value)

        if constants.DEBUG_VERBOSE:
            print "score updated for {}, class {}".format(word_token, class_value)
        score += log2(p)

    score += log2(vocab.get_class_proportion(class_value))
    return score


def read_file_to_lines(file_dir, file_name):

    file_lines = open("{}/{}".format(file_dir, file_name), 'r').readlines()
    if LIMIT_LINES_TO_10:
        file_lines = file_lines[0:20]
    return file_lines


def lines_to_vocab(file_lines):
    vocab = Vocabulary()
    if constants.DEBUG_VERBOSE:
        for index, line in enumerate(file_lines):
            print "{}: {}".format(index, line)

# TODO: Extract method from here
    # remove punctuations from input
    if SANITIZE_LINES:
        file_lines = list(map(sanitize_line_remove_punctuation, file_lines))
    for index, line in enumerate(file_lines):
        if constants.DEBUG_VERBOSE:
            print "line number: {}".format(index)
        line_class_value, line_word_tokens = get_classvalue_and_wordtokens(line)

        if constants.DEBUG_VERBOSE:
            print "class value for line {}: {}".format(line_word_tokens, line_class_value)
#Extract method till here
        for word_occurrence in line_word_tokens:
            vocab.add_word_to_vocabulary(word_occurrence, line_class_value)
    return vocab


def get_classvalue_and_wordtokens(line):
    if SANITIZE_LINES:
        line = sanitize_line_remove_punctuation(line)
    tokens_and_class_value = line.split()
    line_class_value = tokens_and_class_value[len(tokens_and_class_value) - 1]
    line_class_value = int(line_class_value)
    # truncate the class value from line
    line_word_tokens = tokens_and_class_value[0:(len(tokens_and_class_value) - 1)]
    # convert all words to lowercase
    if SANITIZE_TOKENS:
        line_word_tokens = list(map(sanitize_token_to_lowercase, line_word_tokens))
    return line_class_value, line_word_tokens
#
#
# if DEVE
# total_time_spent_in_vocab = 0

def create_classifier_and_classify_naive_bayes(training_set_lines, test_line, smoothing_factor):
    training_vocab = lines_to_vocab(training_set_lines)

    _, word_tokens = get_classvalue_and_wordtokens(test_line)

    if SANITIZE_TOKENS:
        word_tokens = map(sanitize_token_to_lowercase, word_tokens)

    class_scores = []
    for class_i in output_class.CLASSES:
        class_i_score = calculate_score_of_testline_given_class(word_tokens, class_i, training_vocab,
                                                                    smoothing_factor)
        class_scores.append(class_i_score)

    # NOTE: In case of multiple maximums, this function returns the first position.
    index_of_highest_score = class_scores.index(max(class_scores))
    if constants.DEBUG_VERBOSE:
        print "class scores are {} and index {} was selected".format(class_scores, index_of_highest_score)

    return index_of_highest_score


def is_classification_correct(predicted_class_value, actual_class_value):
    return predicted_class_value == actual_class_value


def unit_tests(vocab):
    vocab.unit_test_no_0_0()


def separate_lines_by_class(lines):
    # separated lines is an array of arrays. each element of separated_lines at index i is the list lines from
    # original lines, and all of them have class_value of i
    separated_lines = []
    for _ in output_class.CLASSES:
        separated_lines.append([])


    for line in lines:
        line_classvalue, _ = get_classvalue_and_wordtokens(line)
        separated_lines[line_classvalue].append(line)

    return separated_lines




#  [[part1_lines], [part2_lines], ..., [part_k_lines]]
def stratify_lines_to_k_parts(all_original_lines, k):

    # [[class0_lines], [class1_lines], ...  ]
    lines_partitioned_by_class = separate_lines_by_class(all_original_lines)

    # [number of class1 lines per strata, number of class2 lines per strata, ...]
    numbers_of_classlines_per_part = [len(classlines)/k for classlines in lines_partitioned_by_class]
    if constants.DEBUG_VERBOSE:
        print  numbers_of_classlines_per_part

    all_k_stratas = []

    for i in range(0, k):
        strata_i = []
        for classvalue in output_class.CLASSES: # NEGATIVE, POSITIVE
            strata_i_start = int(i * numbers_of_classlines_per_part[classvalue])
            strata_i_end = int((i + 1) * numbers_of_classlines_per_part[classvalue])
            strata_i += (lines_partitioned_by_class[classvalue][strata_i_start:strata_i_end])

        # Strata i should not start with all of zero labels followed by all the 1 labels. Hence we shuffle.
        random.shuffle(strata_i)

        all_k_stratas.append(strata_i)

    return all_k_stratas


def get_training_and_test_for_ith_split(k_splits, i):
    # pop() removes and returns the i'th split
    testing_lines = k_splits.pop(i)

    training_lines = []

    # NOTE: there are k-1 splits now in k_splits, and we want them for training
    for strata in k_splits:
        training_lines += strata

    return training_lines, testing_lines


def train_test_and_return_accuracy(training_lines, test_lines, smoothing_factor):
    number_of_correct_predictions = 0
    for test_line in test_lines:
        prediction = create_classifier_and_classify_naive_bayes(training_lines, test_line, smoothing_factor)
        actual, _ = get_classvalue_and_wordtokens(test_line)
        if is_classification_correct(prediction, actual):
            number_of_correct_predictions += 1
    return 100 * (number_of_correct_predictions / len(test_lines))


set_environment()


def prepare_k_stratified_train_test_sets(input_file_dir, input_file_name):
    all_original_lines = read_file_to_lines(input_file_dir, input_file_name)
    # FIXME: Do we need to shuffle in the beginning?
    random.shuffle(all_original_lines)
    k_parts = stratify_lines_to_k_parts(all_original_lines, K)

    #  [(train_set_1, test_set_1), ..., [train_set_k, test_set_k]] each set is a list of lines
    train_test_sets = []
    for i in range(0, K):
        #  I had to add this ugly line, otherwise python would pass my list by reference
        pass_k_parts_by_value = k_parts[:]

        train_set_i, test_set_i = get_training_and_test_for_ith_split(pass_k_parts_by_value, i)
        if constants.DEBUG_VERBOSE:
            print "train set {} len {}:\n{}".format(i, len(train_set_i), train_set_i)
            print "test set {} len {}:\n{}\n".format(i, len(test_set_i), test_set_i)
        train_test_sets.append((train_set_i, test_set_i))
    return train_test_sets


# def cross_validate_kfold(train_test_sets):
#     train = 0
#     test = 1
#     accuracies = []
#     for i in range(0, K):
#         if constants.DEBUG_VERBOSE:
#             print "testing on strata {}".format(i+1)
#         accuracy = train_test_and_return_accuracy(train_test_sets[i][train], train_test_sets[i][test], DEFAULT_SMOOTHING_FACTOR)
#         accuracies.append(accuracy)
#
#     std, mean = statistics.calculate_std_mean(accuracies)
#
#     return std, mean


def create_partial_training_sets_and_kfold_crossvalidate_return_accuracies(k_stratified_train_test_set_tuples,
                                                                           smoothing_factor):
    k = len(k_stratified_train_test_set_tuples)
    if constants.DEBUG_VERBOSE:
        print "number of training sets: {}".format(k)
    all_accuracies = []
    for i, train_test_split in enumerate(k_stratified_train_test_set_tuples):
        if constants.DEBUG_CLIENT:
            print "** validating split {} **".format(i)
        test_set_for_kth_split = train_test_split[TEST]
        partial_accuracies = []
        # fractions 0.1, 0.2, 0.3, ..., 1.0
        for i in range(0, 10):
            fraction = (i / 10) + 0.1
            fraction_start = 0
            fraction_end = (int(fraction * len(train_test_split[TRAIN])))

            if constants.DEBUG_CLIENT:
                print "* validating fraction {} *".format(fraction)

            # NOTE: watch the one with 719 fraction end
            if fraction_end == 719:
                fraction_end += 1
            if constants.DEBUG_VERBOSE:
                print "fraction_start: {}, fraction_end: {}".format(fraction_start, fraction_end)

            partial_fraction_train_set = train_test_split[TRAIN][fraction_start:fraction_end]

            partial_accuracy = train_test_and_return_accuracy(partial_fraction_train_set, test_set_for_kth_split,
                                                              smoothing_factor)
            partial_accuracies.append(partial_accuracy)

        all_accuracies.append(partial_accuracies)

    return all_accuracies


def kfold_crossvalidate_for_smoothing_factors_and_return_accuracies(k_stratified_train_test_set_tuples,
                                                                    smoothing_factors):
    k = len(k_stratified_train_test_set_tuples)
    if constants.DEBUG_VERBOSE:
        print "number of training sets: {}".format(k)
    all_splits_accuracies = []
    for i, train_test_split in enumerate(k_stratified_train_test_set_tuples):
        if constants.DEBUG_CLIENT:
            print "** validating split {} **".format(i)
        test_set_for_kth_split = train_test_split[TEST]
        accuracies_for_one_factor = []

        for smoothing_factor in smoothing_factors: # 0, 0.1, ..., 1.0, 2, ..., 20
            if constants.DEBUG_CLIENT:
                print "* validating for smoothing factor {} *".format(smoothing_factor)

            accuracy_for_one_factor = train_test_and_return_accuracy(train_test_split[TRAIN], test_set_for_kth_split,
                                                                     smoothing_factor)
            accuracies_for_one_factor.append(accuracy_for_one_factor)

        all_splits_accuracies.append(accuracies_for_one_factor)

    return all_splits_accuracies

def solution_to_part_one():
    print "starting hw2 part 1"

    start_time = time.time()

    part_1_smoothing_factors = [0, 1]

    for data_set_file_name in DATASET_FILES:
        subplotable_objects = []
        if constants.DEBUG_CLIENT:
            print "**** Starting experiment with data from file: {} ****".format(data_set_file_name)

        for smoothing_factor in part_1_smoothing_factors:
            if constants.DEBUG_CLIENT:
                print "*** Smoothing Factor: {} ***".format(smoothing_factor)
            # returns [(std_0.1, mean_0.1), (std_0.2, mean_0.2), ..., (std_1.0, mean_1.0)]
            k_stratified_train_test_set_tuples = prepare_k_stratified_train_test_sets(
                INPUT_FILES_DIR, data_set_file_name)

            if constants.DEBUG_CLIENT:
                print "data stratified!"

            # HINT: acc_s1p2 = accuracy of split 1, partial fraction 0.2
            # returns: [[acc_s1p1, acc_s1p2, ... acc_s1p10], [acc_s2p1, acc_s2p2, ... acc_s2p10], ...
            # ..., [acc_skp1, acc_skp2, ... acc_skp10]]
            accuracies = create_partial_training_sets_and_kfold_crossvalidate_return_accuracies(
                k_stratified_train_test_set_tuples, smoothing_factor)

            stds, means = statistics.calculate_std_mean(accuracies)

            if constants.DEBUG_VERBOSE:
                print "stds = {},\nmeans = {}".format(stds, means)

            bar_legend_label =  "smoothing factor: {} ".format(smoothing_factor)
            x_values = [90, 180, 270, 360, 450, 540, 630, 720, 810, 900]

            subplotable_object = subplotable.SubPlotable(bar_legend_label, x_values, means, stds)
            subplotable_objects.append(subplotable_object)

        label_plot = "Learning Curves with Cross Validation for {}".format(data_set_file_name.replace(".txt", ""))
        label_plot_x = "Training Size"
        label_plot_y = "Accuracy"
        plot.plot_accuracies_with_stderr_poly(label_plot, label_plot_x, label_plot_y,
                                              [0, 1000], [0, 100], subplotable_objects)

    elapsed_time = time.time() - start_time
    if constants.DEBUG_CLIENT:
        print "program took {} seconds to run".format(elapsed_time)



def solution_to_part_two():
    print "starting hw2 part 2"

    start_time = time.time()

    part_2_smoothing_factors = SMOOTHING_VALUES

    for data_set_file_name in DATASET_FILES:
        subplotable_objects = []
        if constants.DEBUG_CLIENT:
            print "**** Starting experiment with data from file: {} ****".format(data_set_file_name)

        # returns [(std_0.1, mean_0.1), (std_0.2, mean_0.2), ..., (std_1.0, mean_1.0)]
        k_stratified_train_test_set_tuples = prepare_k_stratified_train_test_sets(
            INPUT_FILES_DIR, data_set_file_name)

        if constants.DEBUG_CLIENT:
            print "data stratified!"

        # HINT: acc_s1p2 = accuracy of split 1, partial fraction 0.2
        # returns: [[acc_s1p1, acc_s1p2, ... acc_s1p10], [acc_s2p1, acc_s2p2, ... acc_s2p10], ...
        # ..., [acc_skp1, acc_skp2, ... acc_skp10]]
        accuracies_sets = kfold_crossvalidate_for_smoothing_factors_and_return_accuracies(k_stratified_train_test_set_tuples,
                                                                                     part_2_smoothing_factors)

        stds, means = statistics.calculate_std_mean(accuracies_sets)

        if constants.DEBUG_VERBOSE:
            print "stds = {},\nmeans = {}".format(stds, means)

        bar_legend_label =  "accuracies"
        x_values = part_2_smoothing_factors
        subplotable_object = subplotable.SubPlotable(bar_legend_label, x_values, means, stds)
        subplotable_objects.append(subplotable_object)

        label_plot = "Learning Curves with Cross Validation for {}".format(data_set_file_name.replace(".txt", ""))
        label_plot_x = "Smoothing Factor"
        label_plot_y = "Accuracy"
        plot.plot_accuracies_with_stderr_poly(label_plot, label_plot_x, label_plot_y,
                                              [0, 10], [0, 100], subplotable_objects, "hw2_part2")

    elapsed_time = time.time() - start_time
    if constants.DEBUG_CLIENT:
        print "program took {} seconds to run".format(elapsed_time)



solution_to_part_one()
solution_to_part_two()

