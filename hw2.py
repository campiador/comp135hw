# hw2.py
# by Behnam Heydarshahi, October 2017
# Empirical/Programming Assignment 2
# COMP 135 Machine Learning
#
# This code implements Naive Bayes algorithm with smoothing
# We then evaluate it using cross validation on the task of sentiment analysis

from vocabulary import Vocabulary
from word import Word
import re

DEBUG_DEVELOPER = 1
DEBUG_VERBOSE = 0
ON_SERVER = 0

DATASET_FILE_YELP = "yelp_labelled.txt"


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


def read_file_to_vocab(file_dir, file_name):
    vocab = Vocabulary()

    file_lines = open("{}/{}".format(file_dir, file_name), 'r').readlines()
    # file_lines = file_lines[0:10]

    if DEBUG_VERBOSE:
        for index, line in enumerate(file_lines):
            print "{}: {}".format(index, line)

    # remove punctuations from input
    file_lines = list(map(sanitize_line, file_lines))

    for index, line in enumerate(file_lines):
        if DEBUG_DEVELOPER:
            print "line number: {}".format(index)
        tokens_and_class_value = line.split()

        line_class_value = tokens_and_class_value[len(tokens_and_class_value)-1]
        line_class_value = int(line_class_value)

        # truncate the class value from line
        line_word_tokens = tokens_and_class_value[0:(len(tokens_and_class_value)-1)]
        line_word_tokens = list(map(sanitize_token, line_word_tokens))

        if DEBUG_VERBOSE:
            print "class value for line {}: {}".format(line_word_tokens, line_class_value)

        for word_occurrence in line_word_tokens:
            vocab.add_word_to_vocabulary(word_occurrence, line_class_value)

    # vocab = Vocabulary()

read_file_to_vocab(INPUT_FILES_DIR, DATASET_FILE_YELP)

# w = Word("h", 0, 100)
