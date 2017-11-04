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

from numpy.ma import log2

import constants
import output_class
from word_in_vocab import WordInVocab

SKIP_IT = -1

# CONTINUE HERE:
# TODO: Should I shuffle the vocabulary or do I need another DS for 10-fold stratified cross validation? If the former:
# TODO: Use ordered dict so I can shuffle.
# from collections import OrderedDict
#
# d = OrderedDict()

SMOOTHING_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Vocabulary:
    def __init__(self):
        self.words = {}
        self.total_count = 0
        self.total_count_per_class = []

        for _ in output_class.CLASSES:
            # init to zero
            self.total_count_per_class.append(0)
        # self.words.setdefault(-1)

    def add_word_to_vocabulary(self, word_key, word_class_value):
        if word_key not in self.words:  # first time we see this word
            class_counts = output_class.get_new_output_class_array()
        else:  # we have seen this word before
            word = self.words[word_key]
            class_counts = word.get_class_counts()
        class_counts[word_class_value] += 1
        self.words[word_key] = WordInVocab(class_counts)


    def lookup_total_count_for_class(self, class_number):
        return self.total_count_per_class[class_number]

    def set_total_count_for_class(self, class_number):
        class_count = 0
        for key, word_in_vocab in self.words.items():
            counts = word_in_vocab.get_class_counts()
            class_count += counts[class_number]
        self.total_count_per_class[class_number] = class_count

    def get_word_count_given_class(self, word_key, class_number):
        word_in_vocab = self.words.get(word_key)

        if word_in_vocab is None:
            # FIXME: how are you gonna handle the SKIP_IT situation
            #  Note: Word does not exist in vocabulary at all (neither of classes). Algorithm should skip it.
            return SKIP_IT
        # otherwise
        class_counts = word_in_vocab.get_class_counts()
        return class_counts[class_number]

    def get_class_proportion(self, class_value):

        class_count = self.total_count_per_class[class_value]
        total_count = self.get_total_word_token_count_regardless_of_class()
        return class_count / total_count

    # Number of unique words in vocabulary
    def get_vocabulary_size(self):
        return len(self.words)

    # Total number of tokens in the original text
    def get_total_word_token_count_regardless_of_class(self):
        return self.total_count    # Total number of tokens in the original text

    def set_total_word_token_count_regardless_of_class(self):
        self.total_count = 0
        for word_in_voc in self.words.values():
            class_values = word_in_voc.get_class_counts()
            for class_value in class_values:
                self.total_count += class_value
        return self.total_count

    def unit_test_no_0_0(self):
        for count_values in iter(self.words.values()):
            if count_values[0] == 0 and count_values[1] == 0:
                print "ERROR: found a word in vocabulary with both class counts equal to 0"


    def calculate_probability_of_word_given_class(self, word, class_value, smoothing_factor):
        n_w_in_c = self.get_word_count_given_class(word, class_value)
        if n_w_in_c == SKIP_IT:
            return SKIP_IT
        if n_w_in_c == 0 and constants.DEBUG_VERBOSE:
            print "zero count for {} in class {}".format(word, class_value)
        n_c = self.lookup_total_count_for_class(class_value)
        v = self.get_vocabulary_size()
        m = smoothing_factor
        if constants.DEBUG_VERBOSE:
            print "(n_w_in_c + m)/(n_c + m*v) = ({} + {})/({} + {}*{}) = {}"\
                .format(n_w_in_c, m, n_c, m, v, (n_w_in_c + m)/(n_c + m*v))
        return (n_w_in_c + m)/(n_c + m*v)


    def calculate_score_of_word_given_class(self, word_token, class_value, vocab, smoothing_factor):
        if constants.DEBUG_VERBOSE:
            print "getting score for {}, class {}".format(word_token, class_value)

        p = self.calculate_probability_of_word_given_class(word_token, class_value, smoothing_factor)
        score = log2(p)

        if p == SKIP_IT:
            if constants.DEBUG_VERBOSE:
                print "Skipping {} in class {}".format(word_token, class_value)
            return None
        if constants.DEBUG_VERBOSE and p == 0:
            print "Logging Zero score for {}".format(word_token, class_value)

        if constants.DEBUG_VERBOSE:
            print "score updated for {}, class {}".format(word_token, class_value)
        return score

    def calculate_and_store_score_of_all_words(self):
        for word_key, word_value in self.words.iteritems():
            scores_across_smoothing_values_and_cross_values= []
            for m in SMOOTHING_VALUES:
                class_scores = []
                for class_label in output_class.CLASSES:
                    score_c = self.calculate_score_of_word_given_class(word_key, class_label, self, m)
                    class_scores.append(score_c)
                scores_across_smoothing_values_and_cross_values.append(class_scores)
            word_value.set_scores(scores_across_smoothing_values_and_cross_values)

    def lookup_score_for_word(self, word_key, smoothing, class_value):
        word = self.words[word_key]
        scores = word.get_scores()
        return scores[smoothing][class_value]

    def setup(self):
        for class_number in output_class.CLASSES:
            self.set_total_count_for_class(class_number)
        self.set_total_word_token_count_regardless_of_class()
        self.calculate_and_store_score_of_all_words()
