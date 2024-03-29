#  output_class.py
#  by Behnam Heydarshahi, October 2017
#  Empirical/Programming Assignment 2
#  COMP 135 Machine Learning
#
#  Modifying this class is enough if we want to have more than two class-labels in the future

CLASS_NEGATIVE = 0
CLASS_POSTIVE = 1
CLASSES = [CLASS_NEGATIVE, CLASS_POSTIVE]

class OutputClass:
    def __init__(self, label, value):
        self.label = label
        self.value = value
        return


#  length of array is NUMBER_OF_OUTPUT_CLASSES
def get_new_output_class_array():
    return [0] * len(CLASSES)
