#  statistics.py
#  by Behnam Heydarshahi, October 2017
#  Empirical/Programming Assignment 2
#  COMP 135 Machine Learning
#
#  This class keeps statistical formulas for mean and std deviation
import numpy

import constants


# credit: https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list
# So when you send in an array of arrays, all the first element of the inner arrays will be reduced to two elements in
# the output arrays: one element for stds and one for means.
# Here's an example:
# INPUT:
# A_rank = [0.8, 0.4, 1.2, 3.7, 2.6, 5.8]
# B_rank = [0.1, 2.8, 3.7, 2.6, 5, 3.4]
# C_Rank = [1.2, 3.4, 0.5, 0.1, 2.5, 6.1]
# OUTPUT:
# means: array([0.7, 2.2, 1.8, 2.13333333, 3.36666667, 5.1])
# stds: array([0.45460606, 1.29614814, 1.37355985, 1.50628314, 1.15566239, 1.2083046])

def calculate_std_mean(array_of_arrays):

    arr = numpy.array(array_of_arrays)

    if constants.DEBUG_VERBOSE:
        print "numpy.array:{}".format(arr)

    means = numpy.mean(arr, axis=0)
    stds = numpy.std(arr, axis=0)

    return stds, means
