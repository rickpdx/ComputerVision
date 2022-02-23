import numpy as np
import math
from sklearn.metrics import pairwise_distances


class kMeans:

    def __init__(self, k, seed=None, max_iter=300):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter

    # Initialize the centroids randomly chosen from data points
    # Return centroids
    def initialize_centroids(self, data):
        initial_centroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[initial_centroids]

        return self.centroids

    # Calculates the squared Euclidean Distance, assigns labels, and calculates SSE
    # Return labels
    def assign_clusters(self, data):

        # Check data dimension and reshape if necessary
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # calculate squared Euclidean distance
        if data.ndim == 2:
            distance = pairwise_distances(
                data, self.centroids, metric='sqeuclidean')

        # find arg min and assign to cluster labels
        self.cluster_labels = np.argmin(distance, axis=1)

        # calculate SSE by taking the sum of the distances
        minlist = []
        for i in range(len(self.cluster_labels)):
            minlist.append(distance[i, self.cluster_labels[i]])

        self.SSE = sum(minlist)

        return self.cluster_labels

    # Compute the mean of all data points in a cluster and assigns new centroids
    # Returns centroids
    def update_centroids(self, data):
        # Use cluster labels for updating
        self.centroids = np.array(
            [data[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])

        return self.centroids

    # Model prediction for which cluster the data points are assigned to
    # Returns cluster labels
    def predict(self, data):
        return self.assign_clusters(data)

    # Fits the model by initializing the centroids, assigning clusters, and then updating the centroids
    # Returns instance of the class
    def fit_kmeans(self, data):
        # initialize the centroids
        self.centroids = self.initialize_centroids(data)

        # Main kmeans loop
        for i in range(self.max_iter):
            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
        return self
