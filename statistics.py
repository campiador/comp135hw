# credit: https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list
import numpy

import constants


def calculate_std_mean(experiments):
    # A_rank = [0.8, 0.4, 1.2, 3.7, 2.6, 5.8]
    # B_rank = [0.1, 2.8, 3.7, 2.6, 5, 3.4]
    # C_Rank = [1.2, 3.4, 0.5, 0.1, 2.5, 6.1]

    arr = numpy.array(experiments)

    if constants.DEBUG_VERBOSE:
        print "numpy.array:{}".format(arr)

    mean = numpy.mean(arr, axis=0)

    # array([0.7, 2.2, 1.8, 2.13333333, 3.36666667,
    #        5.1])

    std = numpy.std(arr, axis=0)

    # array([0.45460606, 1.29614814, 1.37355985, 1.50628314, 1.15566239,
    #        1.2083046])
    return std, mean
