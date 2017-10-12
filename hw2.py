# hw2.py
# by Behnam Heydarshahi, October 2017
# Empirical/Programming Assignment 2
# COMP 135 Machine Learning
#
# This code implements Naive Bayes algorithm with smoothing
# We then evaluate it using cross validation on the task of sentiment analysis

from __future__ import division
from vocabulary import Vocabulary
from word import Word
import re
from math import log


# CONSTANTS
CLASS_NEGATIVE = 0
CLASS_POSITIVE = 1

DATASET_FILE_YELP = "yelp_labelled.txt"
DATASET_FILE_IMDB = "imdb_labelled.txt"
DATASET_FILE_AMAZON = "amazon_cells_labelled.txt"

# CONTROL VARIABLES
DEBUG_DEVELOPER = 1
DEBUG_VERBOSE = 0
ON_SERVER = 0

SANITIZE_LINES = 1
SANITIZE_TOKENS = 1

SMOOTHING_FACTOR = 1


def set_environment():
    global INPUT_FILES_DIR
    # Don't put a "/" in the end of the path
    if ON_SERVER:
        INPUT_FILES_DIR = "."
    else:  # On local machine
        INPUT_FILES_DIR = "./sentiment labelled sentences"


# Credit: https://stackoverflow.com/questions/10017147/removing-a-list-of-characters-in-string
def sanitize_line(line):
    chars_to_remove = [".", ",", "!", "?"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    return re.sub(rx, '', line)


def sanitize_token(token):
    return token.lower()


def probability_of_word_given_class(word, class_value, vocab, smoothing_factor):
    n_w_in_c = vocab.get_word_count_given_class(word, class_value)
    n_c = vocab.get_total_count_for_class(class_value)
    v = vocab.get_vocabulary_size()
    m = smoothing_factor
    print "(n_w_in_c + m)/(n_c + m*v) = ({} + {})/({} + {}*{}) ".format(n_w_in_c, m, n_c, m, v)
    return (n_w_in_c + m)/(n_c + m*v)


def read_file_to_vocab(file_dir, file_name):
    vocab = Vocabulary()

    file_lines = open("{}/{}".format(file_dir, file_name), 'r').readlines()
    # file_lines = file_lines[0:10]

    if DEBUG_VERBOSE:
        for index, line in enumerate(file_lines):
            print "{}: {}".format(index, line)

    # remove punctuations from input
    if SANITIZE_LINES:
        file_lines = list(map(sanitize_line, file_lines))

    for index, line in enumerate(file_lines):
        if DEBUG_DEVELOPER:
            print "line number: {}".format(index)
        tokens_and_class_value = line.split()

        line_class_value = tokens_and_class_value[len(tokens_and_class_value)-1]
        line_class_value = int(line_class_value)

        # truncate the class value from line
        line_word_tokens = tokens_and_class_value[0:(len(tokens_and_class_value)-1)]

        # convert all words to lowercase
        if SANITIZE_TOKENS:
            line_word_tokens = list(map(sanitize_token, line_word_tokens))

        if DEBUG_VERBOSE:
            print "class value for line {}: {}".format(line_word_tokens, line_class_value)

        for word_occurrence in line_word_tokens:
            vocab.add_word_to_vocabulary(word_occurrence, line_class_value)

    return vocab


def unit_tests(vocab):
    vocab.unit_test_no_0_0()

set_environment()
vocabulary = read_file_to_vocab(INPUT_FILES_DIR, DATASET_FILE_YELP)
# print vocabulary.get_vocabulary_size()
# print vocabulary.get_total_count_for_class(CLASS_NEGATIVE)
# print vocabulary.get_total_count_for_class(CLASS_POSITIVE)
# print vocabulary.get_word_count_given_class("terrible", CLASS_POSITIVE)
print probability_of_word_given_class("terrible", CLASS_NEGATIVE, vocabulary, SMOOTHING_FACTOR)

unit_tests(vocabulary)


# w = Word("h", 0, 100)
