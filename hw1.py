import commands
import os

from plot import plot_accuracies
from plot import plot_accuracies_with_stderr

import random

import numpy


RUNNING_ON_SERVER = False
CLIENT_DEBUG = False
DEVELOPER_DEBUG = True


CLASSIFIER_J48 = "classifiers.trees.J48"
CLASSIFIER_IBk = "classifiers.lazy.IBk"

DATA_TRAIN_14 = "EEGTrainingData_14.arff"
DATA_TEST_14 = "EEGTestingData_14.arff"
DATA_TRAIN_24 = "EEGTrainingData_24.arff"
DATA_TEST_24 = "EEGTestingData_24.arff"
DATA_TRAIN_34 = "EEGTrainingData_34.arff"
DATA_TEST_34 = "EEGTestingData_34.arff"
DATA_TRAIN_44 = "EEGTrainingData_44.arff"
DATA_TEST_44 = "EEGTestingData_44.arff"
DATA_TRAIN_54 = "EEGTrainingData_54.arff"
DATA_TEST_54 = "EEGTestingData_54.arff"
DATA_TRAIN_64 = "EEGTrainingData_64.arff"
DATA_TEST_64 = "EEGTestingData_64.arff"
DATA_TRAIN_74 = "EEGTrainingData_74.arff"
DATA_TEST_74 = "EEGTestingData_74.arff"
DATA_TRAIN_84 = "EEGTrainingData_84.arff"
DATA_TEST_84 = "EEGTestingData_84.arff"
DATA_TRAIN_94 = "EEGTrainingData_94.arff"
DATA_TEST_94 = "EEGTestingData_94.arff"

all_train_sets = [DATA_TRAIN_14, DATA_TRAIN_24, DATA_TRAIN_34, DATA_TRAIN_44, DATA_TRAIN_54, DATA_TRAIN_64,
                  DATA_TRAIN_74, DATA_TRAIN_84, DATA_TRAIN_94]
all_test_sets = [DATA_TEST_14, DATA_TEST_24, DATA_TEST_34, DATA_TEST_44, DATA_TEST_54, DATA_TEST_64,
                 DATA_TEST_74, DATA_TEST_84, DATA_TEST_94]

set_feature_sizes = [14, 24, 34, 44, 54, 64, 74, 84, 94]

LABEL_J48_ACCURACY = "J48 Accuracy"
LABEL_IBK_ACCURACY = "ibK Accuracy"



def set_environment():
    global CLASSPATH, WEKA_DATA_PATH, WEKA
    if RUNNING_ON_SERVER:
        # For running on Homework server
        # os.environ['CLASSPATH']='/r/aiml/ml-software/weka-3-6-11/weka.jar' + ":" + os.environ['CLASSPATH']
        # os.environ['WEKADATA']='/r/aiml/ml-software/weka-3-6-11/data/'
        if CLIENT_DEBUG:
            print "SHELL: {}".format(os.environ['SHELL'])
        CLASSPATH = os.environ['CLASSPATH']
        WEKA_DATA_PATH = '/r/tcal/work/behnam/135/hw1/data/'
        # os.unsetenv('CLASSPATH:', CLASSPATH + ":{}{}".format(WE))
        if CLIENT_DEBUG:
            print "CLASSPATH: {}".format(CLASSPATH)
            print "wekadata: {}".format(WEKA_DATA_PATH)
        # WEKA = os.getenv('WEKAINSTALL', '/home/behnam/tufts/135/weka-3-9-1/weka.jar')
        # print "weka: {}".format(WEKA)
        WEKA = "weka"

    else:  # Running on local machine
        # WEKA_PATH = os.environ['WEKAINSTALL']
        # WEKA_DATA_PATH = WEKA_PATH + "/data/135/hw1/"

        WEKA_DATA_PATH = '/home/behnam/tufts/135/weka-3-9-1/data/135/'
        CLASSPATH = os.environ['CLASSPATH']
        WEKA_JAR = "/home/behnam/tufts/135/weka-3-9-1/weka.jar"
        CLASSPATH = CLASSPATH + ":{}".format(WEKA_JAR)
        os.environ['CLASSPATH'] = CLASSPATH
        WEKA = "weka"


def test_environment():
    if CLIENT_DEBUG:
        print "CLASSPATH: {}".format(CLASSPATH)
    return


def run_command(str_cmd):
    if CLIENT_DEBUG:
        print "**** RUNNING COMMAND:\n{} ****".format(str_cmd)
    (status, str_output) = commands.getstatusoutput(str_cmd)
    if status != 0:
        print "error in running bash command: {}".format(status)
        print str_output
        exit(2)

    # print str_output

    return str_output


def run_train_test(classifier, train_data, test_data):
    cmd = "java {}.{} -t {}{} -T {}{}".format(WEKA, classifier, WEKA_DATA_PATH, train_data,WEKA_DATA_PATH, test_data)
    return run_command(cmd)


def extract_test_accuracy(str_result):
    i = 0
    lines = str_result.split('\n')
    for line in lines:
        if line.startswith('=== Error on test data ==='):
            accuracy_line = lines[i+2]
            accuracy = accuracy_line.split()[4]
            return accuracy
        i += 1
    print "the output string did not have test accuracy line"
    exit(1)


def run_train_test_return_accuracy(classifier, training, test):
    test_result = run_train_test(classifier, training, test)
    test_accuracy = extract_test_accuracy(test_result)
    return test_accuracy


def run_all_train_test_return_accuracies(classifier):
    accuracies = []
    n_sets = len(all_test_sets)
    for i in range(0, n_sets):
        accuracy = run_train_test_return_accuracy(classifier, all_train_sets[i],all_test_sets[i])
        accuracies.append(accuracy)
    if CLIENT_DEBUG:
        print "accuracies for classifier {}:\n{}".format(classifier, accuracies)
    return accuracies


def solution_to_part_1():
    accuracies_j48 = run_all_train_test_return_accuracies(CLASSIFIER_J48)
    accuracies_ibk = run_all_train_test_return_accuracies(CLASSIFIER_IBk)

    plot_accuracies('Number of Features', 'Test Accuracy', 'Sensitivity to Irrelevant Features',
                    LABEL_J48_ACCURACY, set_feature_sizes, accuracies_j48,
                    LABEL_IBK_ACCURACY, set_feature_sizes, accuracies_ibk)


def remove_file_extension(file_name):

    tokens = file_name.split(".")
    file_name = ""
    extension = ""

    for i, val in enumerate(tokens):
        if i == len(tokens)-1:
            extension = val
        else:
            file_name= file_name + val

    if CLIENT_DEBUG:
        print  "{}{}".format(file_name, extension)

    return file_name, extension


def get_data_threshold(file_name):
    if file_name == DATA_TRAIN_14:
        return 19
    elif file_name == DATA_TRAIN_54:
        return 59
    else:
        print "You need to update data threshold function to support the file {}".format(file_name)
        exit(1)


def shuffle_file_return_header_and_data_rows(file_directory, file_name):
    all_lines = open('{}{}'.format(file_directory, file_name)).readlines()

    threshold = get_data_threshold(file_name)

    header = all_lines[0:threshold]
    data_lines = all_lines[threshold:]

    if CLIENT_DEBUG:
        print "HEADER"
        for line in header:
            print line

    if CLIENT_DEBUG:
        print "LINES:"
        for line in data_lines:
            print line

    random.shuffle(data_lines)

    if CLIENT_DEBUG:
        print "SHUFFLED LINES:"
        for line in data_lines:
            print line

    return header, data_lines


def create_training_set_file(file_directory, file_name, header, lines, training_size_number):
    (file_name_without_extension, file_extension) = remove_file_extension(file_name)

    new_file_name = "{}{}.{}".format(file_name_without_extension, training_size_number, file_extension)

    # only write the required number of training examples
    lines = lines[0: training_size_number]
    if CLIENT_DEBUG:
        print "EXAMPLE SIZE: {}".format(len(lines))

    open('{}{}'.format(file_directory, new_file_name), 'w').writelines(header)
    open('{}{}'.format(file_directory, new_file_name), 'a').writelines(lines)

    return new_file_name




#  credit: https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list
def calculate_std_mean(experiments):
    # A_rank = [0.8, 0.4, 1.2, 3.7, 2.6, 5.8]
    # B_rank = [0.1, 2.8, 3.7, 2.6, 5, 3.4]
    # C_Rank = [1.2, 3.4, 0.5, 0.1, 2.5, 6.1]

    arr = numpy.array(experiments)

    if CLIENT_DEBUG:
        print "numpy.array:{}".format(arr)

    mean = numpy.mean(arr, axis=0)



    # array([0.7, 2.2, 1.8, 2.13333333, 3.36666667,
    #        5.1])

    std = numpy.std(arr, axis=0)



    # array([0.45460606, 1.29614814, 1.37355985, 1.50628314, 1.15566239,
    #        1.2083046])
    return (std, mean)


N_REPETITIONS = 10
N_MAX_TRAINING_SIZE = 500


def get_label_for_experiment(classifier, test_file_name):
    if classifier == CLASSIFIER_J48:
        classifier_label = LABEL_J48_ACCURACY
    elif classifier == CLASSIFIER_IBk:
        classifier_label = LABEL_IBK_ACCURACY

    if test_file_name == DATA_TEST_14:
        feature_label = "14 Features"
    elif test_file_name == DATA_TEST_54:
        feature_label = "54 Features"

    return "{} with {}".format(classifier_label, feature_label)


def solution_to_part_2():
    feature_data = [[DATA_TRAIN_14, DATA_TEST_14], [DATA_TRAIN_54, DATA_TEST_54]]
    classifiers = [CLASSIFIER_J48, CLASSIFIER_IBk]

    four_data_to_be_plotted = []
    for classifier in classifiers:
        for feature in feature_data:
            training_file_name = feature[0]
            test_file_name = feature[1]

            if CLIENT_DEBUG:
                print "\nRunning experiment for {} {} {}\n".format(classifier, training_file_name, test_file_name)

            mean, std = run_experiment_for_classifier_and_feature_size(classifier, training_file_name, test_file_name)

            if CLIENT_DEBUG:
                print  "std{}".format(std)
                print "mean{}".format(mean)
            four_data_to_be_plotted.append([mean, std, classifier, test_file_name])

    plot_accuracies_with_stderr("Learning Curves", "Training Set Size", "Test Accuracy",
                                get_label_for_experiment(four_data_to_be_plotted[0][2], four_data_to_be_plotted[0][3]),
                                list(range(50, N_MAX_TRAINING_SIZE + 50, 50)),
                                four_data_to_be_plotted[0][0], four_data_to_be_plotted[0][1],

                                get_label_for_experiment(four_data_to_be_plotted[1][2], four_data_to_be_plotted[1][3]),
                                list(range(50, N_MAX_TRAINING_SIZE + 50, 50)),
                                four_data_to_be_plotted[1][0], four_data_to_be_plotted[1][1],

                                get_label_for_experiment(four_data_to_be_plotted[2][2], four_data_to_be_plotted[2][3]),
                                list(range(50, N_MAX_TRAINING_SIZE + 50, 50)),
                                four_data_to_be_plotted[2][0], four_data_to_be_plotted[2][1],

                                get_label_for_experiment(four_data_to_be_plotted[3][2], four_data_to_be_plotted[3][3]),
                                list(range(50, N_MAX_TRAINING_SIZE + 50, 50)),
                                four_data_to_be_plotted[3][0], four_data_to_be_plotted[3][1]
                                )


def run_experiment_for_classifier_and_feature_size(classifier, training_file_name, test_file_name):
    iteration_results = []
    for iteration_number in range(1, N_REPETITIONS + 1):  # 1, 2, 3, ..., 10
        if CLIENT_DEBUG:
            print "\nITERATION: {}".format(iteration_number)
            print "shuffling"

        (header, data_rows) = shuffle_file_return_header_and_data_rows(WEKA_DATA_PATH, training_file_name)

        accuracies_vs_training_size = []
        for training_size_number in range(50, N_MAX_TRAINING_SIZE + 50, 50):  # 50, 100, ..., 500
            new_training_file_name = create_training_set_file(WEKA_DATA_PATH, training_file_name,
                                                              header, data_rows, training_size_number)

            accuracy = run_train_test_return_accuracy(classifier, new_training_file_name, test_file_name)
            accuracies_vs_training_size.append(accuracy)

            if CLIENT_DEBUG:
                print "accuracy = {}".format(accuracy)
                print "accuracies = {}".format(accuracies_vs_training_size
                                               )
        accuracies_vs_training_size = map(float, accuracies_vs_training_size)

        iteration_results.append(accuracies_vs_training_size)
    if CLIENT_DEBUG:
        print "\nall {} iteration results:{}".format(N_REPETITIONS, iteration_results)
    (std, mean) = calculate_std_mean(iteration_results)
    return mean, std


set_environment()
test_environment()

solution_to_part_1()
solution_to_part_2()