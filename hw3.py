from math import sqrt

ITERATION_LIMIT = 500
INIT_RANDOM = 0
INIT_SMART = 1

POSITIVE_INFINITY = float("inf")


def has_converged(new_centers):
    if has_converged.last_centers == new_centers:
        return True
    else:
        has_converged.last_centers = new_centers
        return False

has_converged.last_centers = [] # initializing static variable


# TODO: Develop!
def pick_k_cluster_centers(k, examples, random_or_smart_initializtion):
    if INIT_RANDOM == random_or_smart_initializtion:
        pass
    elif INIT_SMART == random_or_smart_initializtion:
        pass
    else:
        print "ERROR: value out of range for random_start: ", random_or_smart_initializtion
        exit(1)


def distance(example, center):
    distance_sum = 0
    center_features = center.features()
    example_features = example.features()

    for i, example_feature in enumerate(example_features):
        distance_sum += (example_feature - center_features[i]) ** 2

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
    clusters = []
    # for i, center in enumerate(centers):
    for example in examples:
        index_closest_center = find_closest_center(example, centers)
        clusters[index_closest_center].append(example)
    pass


def find_center(cluster):
    sum_of_features = 0
    for member in cluster:
        for i, feature_value in member: #FIXME: member data structure unknown
            sum_of_features[i] += feature_value
    cluster_mean  = sum_of_features / len(cluster)

    return cluster_mean



def recalculate_means(clusters):
    new_centers = []
    for cluster in clusters:
        center = find_center(cluster)
        new_centers.append(center)

    return new_centers


def cluster_k_means(k, examples):
    iteration = 0

    centers = pick_k_cluster_centers(k)

    while iteration < ITERATION_LIMIT:
        clusters = associate_examples_with_centers(examples, centers)
        centers = recalculate_means(clusters)
        if has_converged(centers):
            break
        iteration += 1

    return clusters, centers


def parse_examples_from_file_to_list(data_set_file_path):
    pass

# FIXME: NOT SURE how to calculate cluster scatter
def calculate_clustering_scatter(clusters, centers):
    cluster_scatter = 0
    for i, cluster in enumerate(clusters):
        cluster_center = find_center(cluster)
        for cluster_member in cluster:
            for i, attribute in cluster_member:
                # FIXME: assuming each member (or center) is on array of only i attributes
                cluster_scatter += (attribute - cluster_center[i]) ** 2
    return cluster_scatter


# TODO: Implement
def calculate_clustering_nmi(clusters, examples):
    pass


def part_1_1_random_initializtion():

    k = 0
    data_set_file_path = ""

    examples = parse_examples_from_file_to_list(data_set_file_path)

    clusters, centers = cluster_k_means(k, examples)

    cs = calculate_clustering_scatter(clusters, centers)
    nmi = calculate_clustering_nmi(clusters, examples)


def part_1_2_smart_initialization():
    pass


def part_2_effect_of_k_on_cs():
    pass


if __name__ == "__main__":
    part_1_1_random_initializtion()
    part_1_2_smart_initialization()
    part_2_effect_of_k_on_cs()
