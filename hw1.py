import commands
import os

RUNNING_ON_SERVER = False

if RUNNING_ON_SERVER:
    # For running on Homework server
    os.environ['CLASSPATH']='/r/aiml/ml-software/weka-3-6-11/weka.jar' + ":" + os.environ['CLASSPATH']
    os.environ['WEKADATA']='/r/aiml/ml-software/weka-3-6-11/data/'
    WEKA_DATA_PATH = os.environ['WEKADATA']

else:  # Running on local machine
    WEKA_PATH = os.environ['WEKAINSTALL']
    WEKA_DATA_PATH = WEKA_PATH + "/data/135/hw1/"



CLASSIFIER_J48 = ".classifier.trees.J48"



DATA_TRAIN_14 = "EEGTrainingData_14.arff"
DATA_TEST_14 = "EEGTestingData_14.arff"

def check_environment():
    return

def run_command(str_cmd):
    (status, str_output) = commands.getstatusoutput(str_cmd)
    if status != 0:
        print "error in running bash command: {}".format(status)
        print str_output
        exit(1)

    return str_output


def run_train_test(classifier, train_data, test_data):
    return run_command("java weka.{} -t {}\/{} -T {}\/{}").format(classifier, WEKA_DATA_PATH, train_data,
                                                                  WEKA_DATA_PATH, test_data)


run_train_test(CLASSIFIER_J48, DATA_TRAIN_14, DATA_TEST_14)