# example.py
# by Behnam Heydarshahi, November 2017
# Empirical/Programming Assignment 3
# COMP 135 Machine Learning
# modeling example lines:
# each line of arff files include three pieces of information:
#


class Example:
    def __init__(self, id, attributes, label):
        """" @:param id corresponds to line number in the arff file
             @:param attributes corresponds to feature values
             @:param label corresponds to output label
        """
        self.id = id
        self.attributes = attributes
        self.label = label
