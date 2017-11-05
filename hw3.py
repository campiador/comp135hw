# hw3.py
# by Behnam Heydarshahi, November 2017
# Empirical/Programming Assignment 3
# COMP 135 Machine Learning
#
# This code implements k-means algorithm, it stops upon convergence of centers (means) or reaching ITERATION_LIMIT
# PART 1: We initialize centers
#         (i) randomly and then (ii) smartly,
#         We report Cluster Scatter vs Normalized Mutual Information against label-driven golden clustering
# Note: k = #labels
#
# PART 2: We try different K's from 2 to 22, see which one results in better CS.
#


from __future__ import division

import os
import random
from math import sqrt, log

import numpy as np

# Control Variables
from log.log import LOG_VERBOSE, LOG_DEVELOPER
from models.example import data_line_to_example

# from tests.hw3.unit_test import test_hw3_parser

ON_SERVER = False
INPUT_FILES_DIR = "."
ITERATION_LIMIT = 500

# Enums
INIT_RANDOM = 0
INIT_SMART = 1

POSITIVE_INFINITY = float("inf")

DATASET_FILE_ART_05 = "artdata0.5.arff"
DATASET_FILE_ART_1 = "artdata1.arff"
DATASET_FILE_ART_2 = "artdata2.arff"
DATASET_FILE_ART_3 = "artdata3.arff"
DATASET_FILE_ART_4 = "artdata4.arff"
DATASET_FILE_IONOSPHERE = "ionosphere.arff"
DATASET_FILE_IRIS = "iris.arff"
DATASET_FILE_SOYBEAN = "soybean-processed.arff"

DATASETS = [DATASET_FILE_ART_05, DATASET_FILE_ART_1, DATASET_FILE_ART_2, DATASET_FILE_ART_3, DATASET_FILE_ART_4,
            DATASET_FILE_IONOSPHERE, DATASET_FILE_IRIS, DATASET_FILE_SOYBEAN]



def set_environment():
    global INPUT_FILES_DIR
    # Don't put a "/" in the end of the path
    if ON_SERVER:
        INPUT_FILES_DIR = "."
    else:  # On local machine
        INPUT_FILES_DIR = "./input/hw3/data"
    if not os.path.exists("./output/hw3"):
        os.makedirs("./output/hw3")

def has_converged(new_centers):
    if has_converged.last_centers == new_centers:
        return True
    else:
        has_converged.last_centers = new_centers
        return False

has_converged.last_centers = [] # initializing static variable


# NOTE: Add the only the example's features to the centers, not the whole example
def pick_k_cluster_centers(k, examples, random_or_smart_initializtion):
    centers = []
    count_examples = len(examples)
    if INIT_RANDOM == random_or_smart_initializtion:
        while len(centers) < k:
            random_id = random.randrange(0, count_examples)
            random_example = examples[random_id]
            if random_example.features not in centers:
                centers.append(random_example.features)
        return centers
    elif INIT_SMART == random_or_smart_initializtion:
        pass
    else:
        print "ERROR: value out of range for random_start: ", random_or_smart_initializtion
        exit(1)


def distance(example, center):
    distance_sum = 0
    example_features = example.features

    for i, example_feature in enumerate(example_features):
        distance_sum += (example_feature - center[i]) ** 2

    return sqrt(distance_sum)



def find_closest_center(example, centers):
    index_min_distance = 0
    min_distance = POSITIVE_INFINITY
    for i, center in enumerate(centers):
        distance_from_center = distance(example, center)
        if distance_from_center < min_distance:
            min_distance = distance_from_center
            index_min_distance = i
    return index_min_distance


def associate_examples_with_centers(examples, centers):
    clusters = [[] for _ in centers]
    for example in examples:
        index_closest_center = find_closest_center(example, centers)
        clusters[index_closest_center].append(example)
    return clusters


def find_center(cluster):
    sum_of_features = []
    for member in cluster:
        for i, feature_value in enumerate(member.features):
            sum_of_features.append(0)
            sum_of_features[i] += feature_value
    cluster_mean = [feature_i / len(cluster) for feature_i in sum_of_features]

    return cluster_mean


# Passing a random example is necessary in case we run into an empty cluster and need a center for it
def recalculate_means(clusters, random_example):
    new_centers = []
    for cluster in clusters:
        if len(cluster) == 0: # Empty cluster
            if LOG_DEVELOPER:
                print "empty cluster found, using random example"
            center = random_example.features
        else:
            center = find_center(cluster)
        new_centers.append(center)
    return new_centers


def cluster_k_means(k, examples):
    iteration = 0

    centers = pick_k_cluster_centers(k, examples, INIT_RANDOM)

    if LOG_VERBOSE:
        print "centers", centers

    while iteration < ITERATION_LIMIT:
        if LOG_VERBOSE:
            print "\niteration:", iteration
        clusters = associate_examples_with_centers(examples, centers)
        if LOG_VERBOSE:
            print "clusters:", clusters

        random_example = examples[random.randrange(0, len(examples))]

        centers = recalculate_means(clusters, random_example)
        # print "new_centers:", centers
        # print "new_centers length:", len(centers)
        if has_converged(centers):
            if LOG_DEVELOPER:
                print "converged on iteration", iteration
            break
        iteration += 1

    return clusters, centers


def calculate_clustering_scatter(clusters, centers):
    cluster_scatter = 0
    for j, cluster in enumerate(clusters):
        for cluster_member in cluster:
            for i, attribute in enumerate(cluster_member.features):
                # FIXME: assuming each member (or center) is on array of only i attributes
                cluster_scatter += (attribute - centers[j][i]) ** 2

    return cluster_scatter


# Question: if an example is found multiple times in a cluster, should it increment the n score twice? probably yes.
# FIXME: Output should be between 0 and 1
def calculate_clustering_nmi(clusters, golden_clusters):
    row_dimension = len(clusters)
    col_dimension = len(golden_clusters)
    # n_total = row_dimension * col_dimension
    n_total = 0
    n_matrix = np.zeros(shape=(row_dimension, col_dimension), dtype=np.int)
    a_vector = np.zeros(shape=row_dimension, dtype=np.int)
    b_vector = np.zeros(shape=col_dimension, dtype=np.int)

    for i, cluster in enumerate(clusters):
        for j, golden_cluster in enumerate(golden_clusters):
            for member in cluster:
                for golden_member in golden_cluster:
                    if member.features == golden_member.features: # Verify: example (member) data strcuture
                        n_matrix[i][j] += 1
                        a_vector[i] += 1
                        b_vector[j] += 1
                        n_total += 1

    # H(U)
    h_clusters = 0
    for a_i in a_vector:
        h_clusters += a_i * log(n_total / a_i)
    h_clusters /= n_total

    # H(V)
    h_golden_clusters = 0
    for b_i in b_vector:
        h_golden_clusters += b_i * log(n_total / b_i)
    h_golden_clusters /= n_total

    # I(U, V)
    i_clusters_and_golden_clusters = 0
    for i in range(0, row_dimension):
        for j in range(0, col_dimension):
            if n_matrix[i][j] == 0:
                continue
            else:
                i_clusters_and_golden_clusters += (n_matrix[i][j] / n_total) * \
                                                  log((n_matrix[i][j] * n_total) / (a_vector[i] * b_vector[j]))

    # NMI(U, V)
    nmi = (2 * i_clusters_and_golden_clusters) / (h_clusters + h_golden_clusters)

    return nmi


def parse_file_to_lines(file_dir, file_name):

    file_lines = open("{}/{}".format(file_dir, file_name), 'r').readlines()
    # if LIMIT_LINES_TO_10:
    #     file_lines = file_lines[0:20]

    if LOG_VERBOSE:
        print "lines read: ", len(file_lines)
    return file_lines


def part_1_1_random_initializtion():

    k = 0
    data_set_file_path = ""

    file_lines = parse_file_to_lines(data_set_file_path)

    clusters, centers = cluster_k_means(k, examples)

    cs = calculate_clustering_scatter(clusters, centers)
    nmi = calculate_clustering_nmi(clusters, examples)


def part_1_2_smart_initialization():
    pass


def part_2_effect_of_k_on_cs():
    pass


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


def find_data_line_position(file_lines):
    for i, line in enumerate(file_lines):
        line = line.lower()
        if line.startswith("@data"):
            return i
    # if @data not found
    print "input file missed @data (case-insensitive) "
    raise ValueError



def extract_examples(file_lines):
    index_data = find_data_line_position(file_lines)
    data_lines = file_lines[index_data + 1:]
    examples = []
    for i, data_line in enumerate(data_lines):
        example = data_line_to_example(data_line, i)
        examples.append(example)
    return examples


def find_all_k_labels_in_examples(k, examples):
    labels = []
    for example in examples:
        if example.label not in labels:
            labels.append(example.label)
        if len(labels) == k: #we have all the labels, no need to go through more examples
            break
    return labels


def calculate_golden_clusters(k, examples):
    golden_clusters = [[] for _ in range(0, k)]

    all_k_labels = find_all_k_labels_in_examples(k, examples)

    for example in examples:
        cluster_number = all_k_labels.index(example.label)
        golden_clusters[cluster_number].append(example)

    return golden_clusters



if __name__ == "__main__":
    set_environment()
    # part_1_1_random_initializtion()
    # part_1_2_smart_initialization()
    # part_2_effect_of_k_on_cs()
    # test_hw3_parser()

    # for file in DATASETS:
    file = DATASET_FILE_SOYBEAN
    if LOG_DEVELOPER:
        print file
    file_lines = parse_file_to_lines(INPUT_FILES_DIR, file)
    k = determine_number_of_classes(file_lines)
    examples = extract_examples(file_lines)
    if LOG_VERBOSE: #To make sure parsing was successful
        print "dataset:", file
        print "number of classes:", k
        print "first line id:", examples[0].id
        print "first line features:", examples[0].features
        print "first line label:", examples[0].label, "\n"
        print "last line id:", examples[-1].id
        print "last line features:", examples[-1].features
        print "last line label:", examples[-1].label

    clusters, centers = cluster_k_means(k, examples)

    cs = calculate_clustering_scatter(clusters, centers)
    print "cs:", cs

    golden_clusters = calculate_golden_clusters(k, examples)
    nmi = calculate_clustering_nmi(clusters, golden_clusters)
    print "nmi:", nmi

    print "\n"
