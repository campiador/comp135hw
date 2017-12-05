# example.py
# by Behnam Heydarshahi, November 2017
# Empirical/Programming Assignment 3
# COMP 135 Machine Learning
# modeling example lines:
# each line of arff files include three pieces of information:
#


class Example:
    def __init__(self, id, features, label):
        """
        create an example
         :param int id: corresponds to line number in the arff file
         :param list features: corresponds to feature values
         :param str label: corresponds to output label
        """

        self.id = id
        self.features = features
        self.label = label

    def __str__(self):
        return "example {}".format(self.id)

    def __repr__(self):
        return "example {}".format(self.id)





def data_line_to_example(data_line, index):

    data_line = data_line.replace(", ", ",")
    features = data_line.split(",")

    float_features = map(float, features[0:-1])
    label = features[-1]
    example = Example(index, float_features, label)

    return example






