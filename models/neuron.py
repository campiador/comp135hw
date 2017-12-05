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

        self.onehot_label = 0

    def __str__(self):
        return "Neuron x:{0:.{2}f}, s:{1:.{2}f}, d:{3:.{2}f}, L:{4:.{2}f}".format(self.output, self.sum_of_node_inputs,
                                                                                  2, self.delta, self.onehot_label)

    def __repr__(self):
        return "Neuron x:{0:.{2}f}, s:{1:.{2}f}, d:{3:.{2}f}, L:{4:.{2}f}".format(self.output, self.sum_of_node_inputs,
                                                                                  2, self.delta, self.onehot_label)

    def set_onehot_label(self, position_of_neuron_in_layer, all_labels, label):
        label = label.replace("\n", "")
        position_of_output_in_labels = all_labels.index(label)
        print position_of_output_in_labels, position_of_neuron_in_layer
        if position_of_output_in_labels == position_of_neuron_in_layer:
            self.onehot_label = 1.0
        else:
            self.onehot_label = 0.0


