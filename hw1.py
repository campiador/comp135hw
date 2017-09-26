import commands
import os

from plot import plot_accuracies

import random

import numpy

RUNNING_ON_SERVER = False
DEBUGGING = True


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
        if DEBUGGING:
            print "SHELL: {}".format(os.environ['SHELL'])
        CLASSPATH = os.environ['CLASSPATH']
        WEKA_DATA_PATH = '/r/tcal/work/behnam/135/hw1/data/'
        # os.unsetenv('CLASSPATH:', CLASSPATH + ":{}{}".format(WE))
        if DEBUGGING:
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
    if DEBUGGING:
        print "CLASSPATH: {}".format(CLASSPATH)
    return


def run_command(str_cmd):
    if DEBUGGING:
        print "**** RUNNING COMMAND:\n{} ****".format(str_cmd)
    (status, str_output) = commands.getstatusoutput(str_cmd)
    if status != 0:
        print "error in running bash command: {}".format(status)
        print str_output
        exit(1)

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
    if DEBUGGING:
        print "accuracies for classifier {}:\n{}".format(classifier, accuracies)
    return accuracies


def solution_to_part_1():
    accuracies_j48 = run_all_train_test_return_accuracies(CLASSIFIER_J48)
    accuracies_ibk = run_all_train_test_return_accuracies(CLASSIFIER_IBk)

    plot_accuracies('Number of Features', 'Test Accuracy', 'Sensitivity to Irrelevant Features',
                    LABEL_J48_ACCURACY, set_feature_sizes, accuracies_j48,
                    LABEL_IBK_ACCURACY, set_feature_sizes, accuracies_ibk)


def shuffle_example(file_directory, file_name, train_set_size):
    lines = open('{}{}'.format(file_directory, file_name)).readlines()
    header = lines[0:20]
    lines = lines[20:]
    random.shuffle(lines)
    (file_name, file_extension) = remove_file_extension(file_name)

    open('{}{}.{}'.format(file_directory, file_name, file_extension), 'w').writelines(header)
    open('{}{}.{}'.format(file_name, file_name, file_extension), 'w').writelines(lines)


def create_train_set_file():
    pass


#  credit: https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list
def calculate_std_mean(experiments):
    arr = numpy.array([A_rank, B_rank, C_rank])

    numpy.mean(arr, axis=0)

    array([0.7, 2.2, 1.8, 2.13333333, 3.36666667,
           5.1])

    numpy.std(arr, axis=0)

    array([0.45460606, 1.29614814, 1.37355985, 1.50628314, 1.15566239,
           1.2083046])


def solution_to_part_2():



    for i in range(1, 11):  # 1, 2, 3, ..., 10
        shuffle_example()

        for j in range(50, 550, 50):  # 50, 100, ..., 500
            create_train_set_file()

            run_train_test_return_accuracy(CLASSIFIER_J48, all_train_sets[i], all_test_sets[i])


    (std, mean) = calculate_std_mean(experiments)




set_environment()
test_environment()


solution_to_part_2()