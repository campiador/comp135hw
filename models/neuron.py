# neuron.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This module models a neural node, aka a neuron


class Neuron():
    def __init__(self):
        """ sum j = sigma wij * xi """
        self.sum_of_node_inputs = 0
        self.output = 0
        self.delta = 0

    def __str__(self):
        return "Neuron x:{0:.{2}f}, s:{1:.{2}f}".format(self.output, self.sum_of_node_inputs, 2)

    def __repr__(self):
        return "Neuron x:{0:.{2}f}, s:{1:.{2}f}".format(self.output, self.sum_of_node_inputs, 2)


