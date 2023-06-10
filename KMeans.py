import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:

    def __init__(self, K=3, max_iters=10, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # taking K number of random samples from the dataset stored in the variable self.X, without replacement and
        # assigns the samples as the initial centroids for the clustering algorithm. The variable "self.centroids"
        # stores the initial centroids chosen from the dataset.
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to the closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    # The code is looping through each cluster in the variable "clusters" and for each sample in that cluster,
    # it assigns the index of that cluster as the label of that sample. The variable "cluster_idx" holds the index of
    # the current cluster, and "cluster" holds the list of samples that belong to that cluster. The variable
    # "sample_index" holds the index of the current sample in the dataset, and "labels[sample_index]" assigns the
    # value of "cluster_idx" as the label of that sample. Finally, the function returns the "labels" array which
    # contains the label of each sample in the dataset
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # it calculates the euclidean distance between the sample and each centroid using the provided
    # euclidean_distance() function, which is a measure of the similarity between two points in space. The method
    # then finds the index of the closest centroid by using numpy's argmin() function to find the index of the
    # minimum distance in the list of distances. The index of the closest centroid is returned, which will be used to
    # assign the sample to the corresponding cluster.
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    # The method takes in the clusters, which are a list of lists of indices of the data points that belong to each
    # cluster. Then, it calculates the mean value of each cluster by taking the mean of the data points in the
    # cluster (using self.X[cluster], where self.X is the data set) along each feature axis (axis=0). The calculated
    # mean values are then assigned as the centroids of each cluster, and returned.
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    # This function checks whether the k-means algorithm has converged or not. It takes in the old centroids and the
    # new centroids as input. It checks if the centroids have changed by comparing the old centroids with the new
    # centroids. It calculates the Euclidean distance between the old centroids and new centroids for all K centroids
    # using the euclidean_distance function.
    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    # Finally, the function shows the plot by calling the plt.show() function.
    #
    # this function plots the current state of the clusters and their centroids, it uses matplotlib
    # library, and it plots the data points of each cluster using different colors and the centroids using x marker in
    # black color.
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

    # For returning the centroids?
    def cent(self):
        return self.centroids
