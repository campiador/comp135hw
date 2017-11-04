# example.py
# by Behnam Heydarshahi, November 2017
# Empirical/Programming Assignment 3
# COMP 135 Machine Learning
# modeling example lines:
# each line of arff files include three pieces of information:
#


class Example:
    def __init__(self, id, features, label):
        """" @:param id corresponds to line number in the arff file
             @:param features corresponds to feature values
             @:param label corresponds to output label
        """
        self.id = id
        self.features = features
        self.label = label


def data_line_to_example(data_line, index):

    data_line = data_line.replace(", ", ",")
    features = data_line.split(",")

    example = Example(index, features[0: -1], features[-1])

    return example
