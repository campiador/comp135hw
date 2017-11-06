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
import time

from log.log import LOG_VERBOSE, LOG_DEVELOPER, LOG_CLIENT
from models.example import data_line_to_example

# from tests.hw3.unit_test import test_hw3_parser

ON_SERVER = False
INPUT_FILES_DIR = "."
ITERATION_LIMIT = 500

# Enums
INIT_RANDOM = 0
INIT_SMART = 1
NUMBER_OF_RANDOM_INITIALIZTIONS = 10


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
def generate_cluster_centers(k, examples, random_or_smart_initializtion):
    if INIT_RANDOM == random_or_smart_initializtion:
        return generate_k_different_random_centers(examples, k)

    elif INIT_SMART == random_or_smart_initializtion:
        if LOG_DEVELOPER:
            print "init smart centers"
        smart_centers = []
        random_first_center = generate_k_different_random_centers(examples, 1)[0]
        smart_centers.append(random_first_center)

        while len(smart_centers) < k:

            ten_random_centers = generate_k_different_random_centers(examples, 10)
            ten_random_centers_min_distances = [POSITIVE_INFINITY for _ in ten_random_centers]

            for i, random_center in enumerate(ten_random_centers):
                for _, smart_center in enumerate(smart_centers):

                    min_distance_from_this_smart_center = distance(random_center, smart_center)

                    if min_distance_from_this_smart_center < ten_random_centers_min_distances[i]:
                        ten_random_centers_min_distances[i] = min_distance_from_this_smart_center

            largest_min_distance = max(ten_random_centers_min_distances)
            index_of_largest_min_distance = ten_random_centers_min_distances.index(largest_min_distance)

            smart_centers.append(ten_random_centers[index_of_largest_min_distance])

        return smart_centers

    else:
        print "ERROR: value out of range for random_start: ", random_or_smart_initializtion
        exit(1)


def generate_k_different_random_centers(examples, k):
    count_examples = len(examples)

    different_centers = []
    while len(different_centers) < k:
        random_id = random.randrange(0, count_examples)
        random_example = examples[random_id]
        if random_example.features not in different_centers:
            different_centers.append(random_example.features)
    return different_centers


def distance(example_features, center):
    distance_sum = 0

    for i, example_feature in enumerate(example_features):
        distance_sum += (example_feature - center[i]) ** 2

    return sqrt(distance_sum)


def find_closest_center(example, centers):
    index_min_distance = 0
    min_distance = POSITIVE_INFINITY
    for i, center in enumerate(centers):
        distance_from_center = distance(example.features, center)
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
def recalculate_means(clusters, random_centers):
    new_centers = []
    for cluster in clusters:
        if len(cluster) == 0: # Empty cluster
            if LOG_DEVELOPER:
                print "empty cluster found, using the first center, then removing it from list"
            center = random_centers[0]
            random_centers.remove(center)
        else:
            center = find_center(cluster)
        new_centers.append(center)
    return new_centers


def cluster_k_means(k, examples, initialization_method):
    iteration = 0

    #  INITIALIZATION
    centers = generate_cluster_centers(k, examples, initialization_method)

    if LOG_VERBOSE:
        print "centers", centers

    while iteration < ITERATION_LIMIT:
        if LOG_VERBOSE:
            print "iteration:", iteration

        # ASSOCIATION
        clusters = associate_examples_with_centers(examples, centers)
        if LOG_VERBOSE:
            print "clusters:", clusters

        # In case of empty clusters we will need up to k random examples
        n_empty_clusters = 0
        for cluster in clusters:
            if len(cluster) == 0:
                n_empty_clusters += 1
        random_centers = generate_cluster_centers(n_empty_clusters, examples, INIT_RANDOM)  # Yes. Should be RANDOM.

        # ADJUSTING CENTERS
        centers = recalculate_means(clusters, random_centers)

        if has_converged(centers):
            if LOG_VERBOSE:
                print "converged after {} iterations".format(iteration)
            break
        iteration += 1

    return clusters, centers


def calculate_clustering_scatter(clusters, centers):
    cluster_scatter = 0
    for j, cluster in enumerate(clusters):
        for cluster_member in cluster:
            for i, attribute in enumerate(cluster_member.features):
                cluster_scatter += (attribute - centers[j][i]) ** 2

    return cluster_scatter


# Question: if an example is found multiple times in a cluster, should it increment the n score twice? probably yes.
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
                i_clusters_and_golden_clusters += n_matrix[i][j] * log((n_matrix[i][j] * n_total) / (a_vector[i] * b_vector[j]))
    i_clusters_and_golden_clusters /= n_total

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
        if len(labels) == k:  # We have all the labels, no need to go through more examples
            break
    return labels


def calculate_golden_clusters(k, examples):
    golden_clusters = [[] for _ in range(0, k)]

    all_k_labels = find_all_k_labels_in_examples(k, examples)

    for example in examples:
        cluster_number = all_k_labels.index(example.label)
        golden_clusters[cluster_number].append(example)

    return golden_clusters


def cluster_and_return_cs_nmi(k, examples, init_method, golden_clusters):
    clusters, centers = cluster_k_means(k, examples, init_method)
    cs = calculate_clustering_scatter(clusters, centers)
    if LOG_VERBOSE:
        print "cs:", cs
    if len(golden_clusters) == 0:  # if we do not need nmi, just return -1 for it
        nmi = -1
    else:
        nmi = calculate_clustering_nmi(clusters, golden_clusters)
    if LOG_VERBOSE:
        print "nmi:", nmi
    print "\n"

    return cs, nmi


def part_1_1_random_initialization(k, examples, golden_clusters):
    if LOG_CLIENT:
        print "\nPART 1.1: effect of random center initializtion on CS and NMI"

    init_method = INIT_RANDOM
    cs_nmi_list = []
    for i in range(0, NUMBER_OF_RANDOM_INITIALIZTIONS):
        if LOG_CLIENT:
            print "random initialization iteration:", i+1
        cs_nmi = cluster_and_return_cs_nmi(k, examples, init_method, golden_clusters)
        cs_nmi_list.append(cs_nmi)

    return cs_nmi_list


def part_1_2_smart_initialization(k, examples, golden_clusters):
    if LOG_CLIENT:
        print "\nPART 1.2: effect of smart center initializtion on CS and NMI"

    cs_nmi = cluster_and_return_cs_nmi(k, examples, INIT_SMART, golden_clusters)
    return cs_nmi


def part_2_effect_of_k_on_cs(examples, golden_clusters):
    if LOG_CLIENT:
        print "\nPART 2: effect of k on CS"

    k_cs_list = []

    for k in range(2, 23):  # 2, 3, 4, ..., 22
        if LOG_CLIENT:
            print "k=", k

        for i in range(0, 10):
            if LOG_CLIENT:
                print "k:{}, iteration:{}".format(k, i+1)
            lowest_cs = POSITIVE_INFINITY
            (cs, _) = cluster_and_return_cs_nmi(k, examples, INIT_RANDOM, golden_clusters)
            if cs < lowest_cs:
                lowest_cs = cs
        best_of_ten = (k, lowest_cs)
        k_cs_list.append(best_of_ten)

    return k_cs_list

if __name__ == "__main__":
    set_environment()
    start_time = time.time()

    for file in DATASETS:
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


        golden_clusters = calculate_golden_clusters(k, examples)


        part_start_time = time.time()
        part_1_1_random_initialization(k, examples, golden_clusters)
        part_elapsed_time = time.time() - part_start_time
        if LOG_CLIENT:
            print "part 1.1 took {} seconds to run".format(part_elapsed_time)

        part_start_time = time.time()
        part_1_2_smart_initialization(k, examples, golden_clusters)
        part_elapsed_time = time.time() - part_start_time
        if LOG_CLIENT:
            print "part 1.2 took {} seconds to run".format(part_elapsed_time)

        part_start_time = time.time()
        part_2_effect_of_k_on_cs(examples, [])
        part_elapsed_time = time.time() - part_start_time
        if LOG_CLIENT:
            print "part 2 took {} seconds to run".format(part_elapsed_time)

        elapsed_time = time.time() - start_time

        if LOG_CLIENT:
            print "program took {} seconds to run".format(elapsed_time)

