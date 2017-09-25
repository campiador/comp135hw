import commands
import os

RUNNING_ON_SERVER = True

if RUNNING_ON_SERVER:
    # For running on Homework server
    # os.environ['CLASSPATH']='/r/aiml/ml-software/weka-3-6-11/weka.jar' + ":" + os.environ['CLASSPATH']
    # os.environ['WEKADATA']='/r/aiml/ml-software/weka-3-6-11/data/'
    print "SHELL: {}".format(os.environ['SHELL'])
    print "CLASSPATH: {}".format(os.environ['CLASSPATH'])
    WEKA_DATA_PATH = '/r/tcal/work/behnam/135/hw1/data/'
    print "wekadata: {}".format(WEKA_DATA_PATH)
    # WEKA = os.getenv('WEKAINSTALL', '/home/behnam/tufts/135/weka-3-9-1/weka.jar')
    # print "weka: {}".format(WEKA)
    WEKA = "weka"

else:  # Running on local machine
    WEKA_PATH = os.environ['WEKAINSTALL']
    WEKA_DATA_PATH = WEKA_PATH + "/data/135/hw1/"



CLASSIFIER_J48 = "classifiers.trees.J48"
CLASSIFIER_IBk = "classifiers.lazy.IBk"



DATA_TRAIN_14 = "EEGTrainingData_14.arff"
DATA_TEST_14 = "EEGTestingData_14.arff"

def check_environment():
    return

def run_command(str_cmd):
    print "******** RUNNING COMMAND: {} ********".format(str_cmd)
    (status, str_output) = commands.getstatusoutput(str_cmd)
    if status != 0:
        print "error in running bash command: {}".format(status)
        print str_output
        exit(1)

    print str_output

    return str_output


def run_train_test(classifier, train_data, test_data):
    cmd = "java {}.{} -t {}{} -T {}{}".format(WEKA, classifier, WEKA_DATA_PATH, train_data,WEKA_DATA_PATH, test_data)
    run_command(cmd)


run_train_test(CLASSIFIER_J48, DATA_TRAIN_14, DATA_TEST_14)
