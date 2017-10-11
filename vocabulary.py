#  vocabulary.py
#  by Behnam Heydarshahi, October 2017
#  Empirical/Programming Assignment 2
#  COMP 135 Machine Learning
#
#  This class keeps records of vocabulary words and their class counts
#
#  Words are stored in a dictionary called self.words = {('key', class_counts[])}
#    where the key is token's name,
#    and the value is an array of count per classes
#    (I used an array for class counts so that I can use this module later if the classes were more than two.)

import output_class


class Vocabulary:
    def __init__(self):
        self.words = {}

    def add_word_to_vocabulary(self, word_key, word_class_value):
        if word_key not in self.words:  # first time we see this word
            class_counts = output_class.get_new_output_class_array()
        else:  # we have seen this word before
            class_counts = self.words[word_key]

        class_counts[word_class_value] += 1
        self.words[word_key] = class_counts

    # TODO
    def get_word_count(self, word_key, class_number):

        return 0


    #  TODO
    def get_total_count_for_class(self, class_number):

        return 0

    # TODO: Number of unique words
    def get_vocabulary_size(self):

        return 0











