# hw4.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This module models a neural node


class Neuron():
    def __init__(self):
        """ sum j = sigma wij * xi """
        self.sum = 0
        self.output = 0
        self.delta = 0
