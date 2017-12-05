# arffparser.py
# by Behnam Heydarshahi, December 2017
# Empirical/Programming Assignment 4
# COMP 135 Machine Learning
#
# This module parses arff files
#
from log.log import LOG_VERBOSE
from models.example import data_line_to_example


def parse_file_to_lines(file_dir, file_name):

    file_lines = open("{}/{}".format(file_dir, file_name), 'r').readlines()

    if LOG_VERBOSE:
        print "lines read: ", len(file_lines)
    return file_lines


def determine_number_of_classes(file_lines):
    class_line = ""
    for file_line in file_lines:
        file_line = file_line.lower()
        if file_line.startswith("@attribute class"):
            class_line = file_line
            break
    if class_line is "":
        raise ValueError
    else: # class_line found
        classes_string = class_line[class_line.index('{') + 1:class_line.index('}')]
        classes_string.replace(" ,", ",")
        classes_list = classes_string.split(",")
        return len(classes_list)


def extract_output_classes(file_lines):
    class_line = ""
    for file_line in file_lines:
        file_line = file_line.lower()
        if file_line.startswith("@attribute class"):
            class_line = file_line
            break
    if class_line is "":
        raise ValueError
    else: # class_line found
        classes_string = class_line[class_line.index('{') + 1:class_line.index('}')]
        classes_string.replace(" ,", ",")
        classes_list = classes_string.split(",")
    for i, label in enumerate(classes_list):
        classes_list[i] = label.replace("\n", "")

    return classes_list

def extract_examples(file_lines):
    index_data = find_data_line_position(file_lines)
    data_lines = file_lines[index_data + 1:]
    examples = []
    for i, data_line in enumerate(data_lines):
        example = data_line_to_example(data_line, i)
        examples.append(example)
    return examples


def find_data_line_position(file_lines):
    for i, line in enumerate(file_lines):
        line = line.lower()
        if line.startswith("@data"):
            return i
    # if @data not found
    print "input file missed @data (case-insensitive) "
    raise ValueError


def parse_file_and_extract_examples_and_number_of_classes_and_features(input_dir, file_name):
    file_lines = parse_file_to_lines(input_dir, file_name)
    examples = extract_examples(file_lines)
    n_features = len(examples[0].features)
    n_classes = determine_number_of_classes(file_lines)
    output_classes = extract_output_classes(file_lines)

    return examples, n_classes, n_features, output_classes


def parse_file_and_extract_examples(input_dir, file_name):
    file_lines = parse_file_to_lines(input_dir, file_name)
    examples = extract_examples(file_lines)

    return examples


