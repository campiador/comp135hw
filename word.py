#  word.py
#  by Behnam Heydarshahi, October 2017
#  Empirical/Programming Assignment 2
#  COMP 135 Machine Learning
#
#  This class keeps records of word tokenss and their class counts

# TODO: make 'class number'-wise polymorphic
class Word:
    def __init__(self, key_name, count_n, count_p):
        self.key_name = key_name
        self.count_n = count_n
        self.count_p = count_p
