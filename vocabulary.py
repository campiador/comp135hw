#  vocabulary.py
#  by Behnam Heydarshahi, October 2017
#  Empirical/Programming Assignment 2
#  COMP 135 Machine Learning
#
#  This class keeps records of vocabulary words and their class counts
#
#  Words and their class_counts are stored in a dictionary called self.words = {('word_key', class_counts[])}
#    where the key is word's name,
#    and the value is an array of count per classes
#    (I used an array for class counts so that I can use this module later if the classes were more than two.)
from __future__ import division

import output_class

SKIP_IT = -1

# CONTINUE HERE:
# TODO: Should I shuffle the vocabulary or do I need another DS for 10-fold stratified cross validation? If the former:
# TODO: Use ordered dict so I can shuffle.
# from collections import OrderedDict
#
# d = OrderedDict()

class Vocabulary:
    def __init__(self):
        self.words = {}
        # self.words.setdefault(-1)

    def add_word_to_vocabulary(self, word_key, word_class_value):
        if word_key not in self.words:  # first time we see this word
            class_counts = output_class.get_new_output_class_array()
        else:  # we have seen this word before
            class_counts = self.words[word_key]

        class_counts[word_class_value] += 1
        self.words[word_key] = class_counts

    def get_total_count_for_class(self, class_number):
        class_count = 0
        for class_values in iter(self.words.values()):
            class_count += class_values[class_number]
        return class_count

    def get_word_count_given_class(self, word_key, class_number):
        class_counts = self.words.get(word_key)
        if class_counts is None:
            # FIXME: how are you gonna handle the SKIP_IT situation
            #  Note: Word does not exist in vocabulary at all (neither of classes). Algorithm should skip it.
            return SKIP_IT
        return class_counts[class_number]

    def get_class_proportion(self, class_value):

        class_count = self.get_total_count_for_class(class_value)
        total_count = self.get_total_word_token_count_regardless_of_class()

        return class_count / total_count

    # Number of unique words in vocabulary
    def get_vocabulary_size(self):
        return len(self.words)

    # Total number of tokens in the original text
    def get_total_word_token_count_regardless_of_class(self):
        total_count = 0
        for class_values in iter(self.words.values()):
            for class_value in class_values:
                total_count += class_value
        return total_count

    def unit_test_no_0_0(self):
        for count_values in iter(self.words.values()):
            if count_values[0] == 0 and count_values[1] == 0:
                print "ERROR: found a word in vocabulary with both class counts equal to 0"

