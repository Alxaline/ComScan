# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
Created on: Jan 14, 2021
"""
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import umap
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

with warnings.catch_warnings():
    # Ignore flood of RuntimeWarning: Explicit initial center position passed: performing only one init
    # in k-means instead of n_init=10 return_n_iter=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from k_means_constrained import KMeansConstrained


def kmeans_constrained_missing(X: Union[pd.DataFrame, np.ndarray], n_clusters: int, size_min: Optional[int] = None,
                               max_iter: int = 10, features_reduction: Optional[str] = None, n_components: int = 2,
                               random_state: Optional[int] = None) \
        -> Tuple[KMeansConstrained, np.ndarray, np.ndarray, np.float, np.ndarray]:
    """
    K-Means ComScan with minimum and maximum cluster size constraints with the possibility of missing values.
    # inspired of https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data

    :param X: array-like or DataFrame of floats, shape (n_samples, n_features)
        The observations to cluster.
    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :param size_min: Constrain the label assignment so that each cluster has a minimum size of size_min.
        If None, no constrains will be applied. default: None
    :param max_iter: Maximum number of EM iterations to perform. default: 10
    :param features_reduction: Method for reduction of the embedded space with n_components. Can be pca or umap.
        Default: None
    :param n_components: Dimension of the embedded space for features reduction. Default 2.
    :param random_state: int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :return:
        labels: label[i] is the code or index of the centroid the i'th observation is closest to.
        centroid: Centroids found at the last iteration of k-means.
        inertia: The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).
        X_hat: Copy of X with the missing values filled in.
    """
    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    if features_reduction is not None:
        assert features_reduction in ["umap", "pca"], "method need to be 'umap' or 'pca'"
        if features_reduction.lower() == "umap":
            X_hat = umap.UMAP(n_components=n_components, random_state=random_state).fit_transform(X_hat)
        elif features_reduction.lower() == "pca":
            X_hat = PCA(n_components=n_components, random_state=random_state).fit_transform(X_hat)
        missing = ~np.isfinite(X_hat)

    cls = None
    prev_labels, labels = np.array([]), np.array([])
    prev_centroids, centroids = np.array([]), np.array([])

    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = KMeansConstrained(n_clusters, init=prev_centroids, size_min=size_min, random_state=random_state)
        else:
            # do multiple random initializations in parallel
            cls = KMeansConstrained(n_clusters, size_min=size_min, random_state=random_state)

        # perform on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break
        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    inertia = cls.inertia_

    return cls, labels, centroids, inertia, X_hat


def optimal_clustering(X: Union[pd.DataFrame, np.ndarray], size_min: int = 10, method: str = "silhouette",
                       features_reduction: Optional[str] = None, n_components: int = 2,
                       random_state: Optional[int] = None) \
        -> Tuple[KMeansConstrained, int, np.ndarray, int, Sequence[np.float], np.float, np.ndarray, np.ndarray]:
    """
    Function to find the optimal clustering using a constrained k means. Two method are available to find the optimal
    number of cluster 'silhouette' and 'elbow'.

    :param X: array-like or DataFrame of floats, shape (n_samples, n_features)
        The observations to cluster.
    :param size_min: Constrain the label assignment so that each cluster has a minimum size of size_min.
        If None, no constrains will be applied. default: None
    :param method: Method to find the optimal number of cluster : "elbow" or "silhouette"
    :param features_reduction: Method for reduction of the embedded space with n_components. Can be pca or umap.
        Default: None
    :param n_components: Dimension of the embedded space for features reduction. Default 2.
    :param random_state: int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :return:
        cls: KMeansConstrained classifier object
        cluster_nb: optimal number of cluster
        labels: label[i] is the code or index of the centroid the i'th observation is closest to.
        centroid: Centroids found at the last iteration of k-means.
        ref_label: cluster label with the minimal within-cluster sum-of-squares.
        wicss_clusters: within-cluster sum-of-squares for each cluster
        best_wicss_cluster: minimal wicss
        inertia: The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).
        X_hat: Copy of X with the missing values filled in.
    """
    assert method in ["elbow", "silhouette"], "method need to be 'elbow' or 'silhouette'"

    n_samples = X.shape[0]
    min_cluster = 2 if method == "silhouette" else 1
    max_cluster = n_samples // size_min + 1

    K = range(min_cluster, max_cluster)

    assert K, ValueError(f"Number of cluster is incompatible with method {method}, min cluster is {min_cluster} and "
                         f"max cluster is {max_cluster - 1} with a size_min of {size_min}")

    if max_cluster - 1 == 1:
        warnings.warn("Only one cluster is possible")

    sil, wcss = [], []
    all_cls, all_labels, all_centroids, all_inertia, all_Xhat = [], [], [], [], []

    for k in K:
        cls, labels, centroids, inertia, X_hat = kmeans_constrained_missing(X,
                                                                            n_clusters=k,
                                                                            size_min=size_min,
                                                                            max_iter=10,
                                                                            features_reduction=features_reduction,
                                                                            n_components=n_components,
                                                                            random_state=random_state)

        all_cls.append(cls), all_labels.append(labels), all_centroids.append(centroids), all_inertia.append(
            inertia), all_Xhat.append(X_hat)

        if method == "elbow":
            wcss.append(inertia)
        elif method == "silhouette":
            sil.append(silhouette_score(X_hat, labels, metric='euclidean'))

    # use knee locator for find the optimal number of cluster
    cluster_nb = 0
    if method == "elbow":
        kn = KneeLocator(K, wcss, curve='convex', direction='decreasing')
        cluster_nb = kn.knee
    elif method == "silhouette":
        sil_score_max = max(sil)
        cluster_nb = K[sil.index(sil_score_max)]

    # get cls, labels, centroids, inertia, Xhat corresponding to cluster_nb
    index = K.index(cluster_nb)
    cls, labels, centroids, inertia, X_hat = all_cls[index], all_labels[index], all_centroids[index], all_inertia[
        index], all_Xhat[index]

    # cluster choose for reference is the one which minimizing a criterion known as the inertia
    # or within-cluster sum-of-squares (WCSS)
    # WCSS is defined as the sum of the squared distance between each member of the cluster and its centroid.
    # WSS=\sum ^{M}_{i=1}\left( x_{i}-c_{i}\right) ^{2}

    # Calculate the distances matrix between all data points and the final centroids.
    D = cdist(X_hat, centroids, 'euclidean')
    wicss_clusters = []
    for label in range(0, cluster_nb):
        labels_bool = labels == label
        # distance to the closest centroid
        dist = D[labels_bool, label]
        # Total with-in sum of square
        wicss_cluster = sum(dist ** 2)
        wicss_clusters.append(wicss_cluster)
        # best is minimal wicss

    best_wicss_cluster = np.min(wicss_clusters)
    ref_label = np.argmin(wicss_clusters)

    return cls, cluster_nb, labels, ref_label, wicss_clusters, best_wicss_cluster, centroids, X_hat
