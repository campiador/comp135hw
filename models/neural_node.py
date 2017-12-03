# hw4.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This module models a neural node


class NeuralNode():
    def __init__(self, sum, output):
        """ sum j = sigma wij * xi """
        self.sum = sum
        self.output = output
