import numpy as np


def db_scan_algorithm(vector_list, minimum_points, epsilon):
    # initially the list of clusters with all the points that are being unclassified
    list_of_cluster_per_point = ([0] * len(vector_list))
    cluster_id = 1
    for i in range(0, len(vector_list)):
        # checking for each point's neighbours and if > minimum points,then classifying it as a core point,
        # adding neighbours to a seed list and expanding cluster by calling expanded cluster,if not classifying the
        # point as noise for now.
        if list_of_cluster_per_point[i] == 0:
            seeds = (finding_neighbour_points(vector_list, i, epsilon, list_of_cluster_per_point, cluster_id))
            if len(seeds) >= minimum_points:
                list_of_cluster_per_point[i] = cluster_id
                expanding_clusters(vector_list, seeds, list_of_cluster_per_point, cluster_id, minimum_points, epsilon)
                cluster_id = cluster_id + 1
            else:
                list_of_cluster_per_point[i] = -1
    # returning the list of clusters after all the points have been travsersed,
    return list_of_cluster_per_point


def finding_neighbour_points(vector_list, iteration, epsilon, cluster_per_point_list, cluster_id):
    neighbours = []
    # Distance calculation for a point to all the other points
    #EUCLIDEAN DISTANCE
    l = np.linalg.norm(np.array(vector_list) - np.array(vector_list[iteration]), axis=1)

    # storing the indices of the point that are neighbours in a list and returning them
    neighbours = [i for i in range(0, len(l)) if l[i] <= epsilon]

    return neighbours


def expanding_clusters(vector_list, seeds, list_of_cluster_per_point, cluster_id, minimum_points, epsilon):
    c = 0
    # expanding the cluster by taking all the neighbours of a core object and checking them for being core objects,
    # if not,just updating their cluster value.If yes,we add their neighbours to the seed list. We do it until there
    # is no untraversed point remaining in seed list
    while c < len(seeds):
        if list_of_cluster_per_point[seeds[c]] == -1:
            list_of_cluster_per_point[seeds[c]] = cluster_id
        elif list_of_cluster_per_point[seeds[c]] == 0:
            list_of_cluster_per_point[seeds[c]] = cluster_id
            new_neighbours = finding_neighbour_points(vector_list, seeds[c], epsilon, list_of_cluster_per_point,
                                                      cluster_id)
            if len(new_neighbours) >= minimum_points:
                seeds = seeds + new_neighbours
        c = c + 1
