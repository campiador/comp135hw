ITERATION_LIMIT = 500


def has_converged():
    False


def pick_k_cluster_centers(k):
    pass


def associate_examples_with_centers():
    pass


def recalculate_means():
    pass


def cluster_k_means(k, examples):
    iteration = 0

    pick_k_cluster_centers(k)

    while iteration < ITERATION_LIMIT:
        associate_examples_with_centers()
        recalculate_means()
        if has_converged():
            break
        iteration += 1





if __name__ == "__main__":
    ""