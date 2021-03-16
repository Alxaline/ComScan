# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 14, 2021
"""
import time
import warnings
from collections.abc import Iterable
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import umap
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.base import clone
from sklearn.decomposition import PCA
from yellowbrick.cluster.elbow import ClusteringScoreVisualizer, KELBOW_SCOREMAP, YellowbrickValueError, \
    YellowbrickWarning
from yellowbrick.style.palettes import LINE_COLOR

with warnings.catch_warnings():
    # Ignore flood of RuntimeWarning: Explicit initial center position passed: performing only one init
    # in k-means instead of n_init=10 return_n_iter=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from k_means_constrained import KMeansConstrained


class KMeansConstrainedMissing(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    K-Means clustering with minimum and maximum cluster size constraints with possible missing values

    .. note::
        inspired of `<https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data>`_

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    size_min : int, optional, default: None
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied

    size_max : int, optional, default: None
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied

    em_iter :  int, default: 10
        expectation–maximization (EM) iteration for convergence of missing
        values. Use when no features reduction is applied and missing values.

    n_init : int, default: 10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    features_reduction : str, default: None
        Method for reduction of the embedded space with n_components. Can be pca or umap.

    n_components : int, default: 2
         Dimension of the embedded space for features reduction.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    verbose : int, default 0
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    cls_ : KMeansConstrained classifier object

    cls_features_reduction_ : PCA or UMAP reduction object

    centroids_: array
        Centroids found at the last iteration of k-means.

    X_hat_ : array
        Copy of X with the missing values filled in.

    mu_ : Columns means

    Examples
    --------

    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...                [4, 2], [4, 4], [4, 0]])
    >>> clf = KMeansConstrainedMissing(
    ...     n_clusters=2,
    ...     size_min=2,
    ...     size_max=5,
    ...     random_state=0
    ... )
    >>> clf.fit_predict(X)
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> clf.cluster_centers_
    array([[ 1.,  2.],
           [ 4.,  2.]])
    >>> clf.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)

    Notes
    ------
    K-means problem constrained with a minimum and/or maximum size for each cluster.

    The constrained assignment is formulated as a Minimum Cost Flow (MCF) linear network optimisation
    problem. This is then solved using a cost-scaling push-relabel algorithm. The implementation used is
     Google's Operations Research tools's `SimpleMinCostFlow`.

    Ref:
    1. Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering."
        Microsoft Research, Redmond (2000): 1-8.
    2. Google's SimpleMinCostFlow implementation:
        https://github.com/google/or-tools/blob/master/ortools/graph/min_cost_flow.h
    """

    def __init__(self, n_clusters=8,
                 size_min=None,
                 size_max=None,
                 em_iter=10,
                 n_init=10,
                 max_iter=300,
                 features_reduction: Optional[str] = None,
                 n_components: int = 2,
                 tol=1e-4,
                 verbose=False,
                 random_state=None,
                 copy_x=True,
                 n_jobs=1):

        self.n_clusters = n_clusters
        self.size_min = size_min
        self.size_max = size_max
        self.em_iter = em_iter
        self.n_init = n_init
        self.max_iter = max_iter
        self.features_reduction = features_reduction
        self.n_components = n_components
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Compute k-means clustering with given constants.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored

        """

        columns_df = []
        if isinstance(X, pd.DataFrame):
            columns_df = list(X.columns)
            if self.features_reduction is not None:
                columns_df = list(map(lambda x: f"Dimension_{x}", list(range(1, self.n_components + 1))))
            X = X.to_numpy()

        # Initialize missing values to their column means
        missing = ~np.isfinite(X)
        self.mu_ = np.nanmean(X, 0, keepdims=1)
        self.X_hat_ = np.where(missing, self.mu_, X)

        self.cls_ = None
        self.cls_features_reduction_ = None
        if self.features_reduction is not None or not np.any(missing):
            if self.features_reduction:
                assert self.features_reduction in ["umap", "pca"], "method need to be 'umap' or 'pca'"
                if self.features_reduction.lower() == "umap":
                    self.cls_features_reduction_ = umap.UMAP(n_components=self.n_components,
                                                             random_state=self.random_state,
                                                             n_jobs=self.n_jobs)
                elif self.features_reduction.lower() == "pca":
                    self.cls_features_reduction_ = PCA(n_components=self.n_components, random_state=self.random_state)

                self.cls_features_reduction_.fit(self.X_hat_)
                self.X_hat_ = self.cls_features_reduction_.transform(self.X_hat_)

            self.cls_ = KMeansConstrained(self.n_clusters,
                                          init="k-means++",
                                          size_min=self.size_min,
                                          random_state=self.random_state,
                                          n_jobs=self.n_jobs)

            self.labels_ = self.cls_.fit_predict(self.X_hat_)
            self.centroids_ = self.cls_.cluster_centers_

        else:
            prev_labels, labels = np.array([]), np.array([])
            prev_centroids, centroids = np.array([]), np.array([])
            for i in range(self.em_iter):
                if i > 0:
                    # initialize KMeans with the previous set of centroids. this is much
                    # faster and makes it easier to check convergence (since labels
                    # won't be permuted on every iteration), but might be more prone to
                    # getting stuck in local minima.
                    self.cls_ = KMeansConstrained(self.n_clusters,
                                                  init=prev_centroids,
                                                  size_min=self.size_min,
                                                  random_state=self.random_state,
                                                  n_jobs=self.n_jobs)
                else:
                    # do multiple random initializations in parallel
                    self.cls_ = KMeansConstrained(self.n_clusters,
                                                  init="k-means++",
                                                  size_min=self.size_min,
                                                  random_state=self.random_state,
                                                  n_jobs=self.n_jobs)

                # perform on the filled-in data
                self.labels_ = self.cls_.fit_predict(self.X_hat_)
                self.centroids_ = self.cls_.cluster_centers_

                # fill in the missing values based on their cluster centroids
                self.X_hat_[missing] = self.centroids_[self.labels_][missing]

                # when the labels have stopped changing then we have converged
                if i > 0 and np.all(self.labels_ == prev_labels):
                    break
                prev_labels = self.labels_
                prev_centroids = self.cls_.cluster_centers_

        self.inertia_ = self.cls_.inertia_
        self.cluster_centers_ = self.cls_.cluster_centers_

        if columns_df:
            self.X_hat_ = pd.DataFrame(self.X_hat_, columns=columns_df)

        return self

    def predict(self, X, size_min='init', size_max='init'):
        """
        Predict the closest cluster each sample in X belongs to given the provided constraints.
        The constraints can be temporally overridden when determining which cluster each datapoint is assigned to.

        Only computes the assignment step. It does not re-fit the cluster positions.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        size_min : int, optional, default: size_min provided with initialisation
            Constrain the label assignment so that each cluster has a minimum
            size of size_min. If None, no constrains will be applied.
            If 'init' the value provided during initialisation of the
            class will be used.

        size_max : int, optional, default: size_max provided with initialisation
            Constrain the label assignment so that each cluster has a maximum
            size of size_max. If None, no constrains will be applied.
            If 'init' the value provided during initialisation of the
            class will be used.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        labels = self.cls_.predict(X, size_min=size_min, size_max=size_max)

        return labels

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Equivalent to calling fit(X) followed by predict(X) but also more efficient.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).cls_.labels_


class KElbowVisualizer(ClusteringScoreVisualizer):
    """
    The K-Elbow Visualizer implements the "elbow" method of selecting the
    optimal number of clusters for K-means clustering. K-means is a simple
    unsupervised machine learning algorithm that groups data into a specified
    number (k) of clusters. Because the user must specify in advance what k to
    choose, the algorithm is somewhat naive -- it assigns all members to k
    clusters even if that is not the right k for the dataset.

    The elbow method runs k-means clustering on the dataset for a range of
    values for k (say from 1-10) and then for each value of k computes an
    average score for all clusters. By default, the ``distortion`` score is
    computed, the sum of square distances from each point to its assigned
    center. Other metrics can also be used such as the ``silhouette`` score,
    the mean silhouette coefficient for all samples or the
    ``calinski_harabasz`` score, which computes the ratio of dispersion between
    and within clusters.

    When these overall metrics for each model are plotted, it is possible to
    visually determine the best value for k. If the line chart looks like an
    arm, then the "elbow" (the point of inflection on the curve) is the best
    value of k. The "arm" can be either up or down, but if there is a strong
    inflection point, it is a good indication that the underlying model fits
    best at that point.

    # yellowbrick.cluster.elbow
    # Implements the elbow method for determining the optimal number of clusters.
    #
    # Author:   Benjamin Bengfort
    # Created:  Thu Mar 23 22:36:31 2017 -0400
    #
    # Copyright (C) 2016 The scikit-yb developers
    # For license information, see LICENSE.txt
    #
    # ID: elbow.py [5a370c8] benjamin@bengfort.com $

    Parameters
    ----------

    estimator : a scikit-learn clusterer
        Should be an instance of an unfitted clusterer, specifically ``KMeans`` or
        ``MiniBatchKMeans``. If it is not a clusterer, an exception is raised.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    k : integer, tuple, or iterable
        The k values to compute silhouette scores for. If a single integer
        is specified, then will compute the range (2,k). If a tuple of 2
        integers is specified, then k will be in np.arange(k[0], k[1]).
        Otherwise, specify an iterable of integers to use as values for k.

    metric : string, default: ``"distortion"``
        Select the scoring metric to evaluate the clusters. The default is the
        mean distortion, defined by the sum of squared distances between each
        observation and its closest centroid. Other metrics include:

        - **distortion**: mean sum of squared distances to centers
        - **silhouette**: mean ratio of intra-cluster and nearest-cluster distance
        - **calinski_harabasz**: ratio of within to between cluster dispersion

    timings : bool, default: True
        Display the fitting time per k to evaluate the amount of time required
        to train the clustering model.

    locate_elbow : bool, default: True
        Automatically find the "elbow" or "knee" which likely corresponds to the optimal
        value of k using the "knee point detection algorithm". The knee point detection
        algorithm finds the point of maximum curvature, which in a well-behaved
        clustering problem also represents the pivot of the elbow curve. The point is
        labeled with a dashed line and annotated with the score and k values.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    k_scores_ : array of shape (n,) where n is no. of k values
        The silhouette score corresponding to each k value.

    k_timers_ : array of shape (n,) where n is no. of k values
        The time taken to fit n KMeans model corresponding to each k value.

    elbow_value_ : integer
        The optimal value of k.

    elbow_score_ : float
        The silhouette score corresponding to the optimal value of k.

    Examples
    --------

    >>> from yellowbrick.cluster import KElbowVisualizer
    >>> from sklearn.cluster import KMeans
    >>> model = KElbowVisualizer(KMeans(), k=10)
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> model.fit(X)
    >>> model.show()

    Notes
    -----

    Modification from yellowbrick consist of get the best_estimator based
    on the finded elbow_value

    If you get a visualizer that doesn't have an elbow or inflection point,
    then this method may not be working. The elbow method does not work well
    if the data is not very clustered; in this case, you might see a smooth
    curve and the value of k is unclear. Other scoring methods, such as BIC or
    SSE, also can be used to explore if clustering is a correct choice.

    For a discussion on the Elbow method, read more at
    `Robert Gove's Block website <https://bl.ocks.org/rpgove/0060ff3b656618e9136b>`_.
    For more on the knee point detection algorithm see the paper `"Finding a "kneedle"
    in a Haystack" <https://raghavan.usc.edu//papers/kneedle-simplex11.pdf>`_.

    .. seealso:: The scikit-learn documentation for the `silhouette_score
        <https://bit.ly/2LYWjYb>`_ and `calinski_harabasz_score
        <https://bit.ly/2ItAgts>`_. The default, ``distortion_score``, is
        implemented in ``yellowbrick.cluster.elbow``.

    .. todo:: add parallelization option for performance
    .. todo:: add different metrics for scores and silhouette
    .. todo:: add timing information about how long it's taking
    """

    def __init__(
            self,
            estimator,
            ax=None,
            k=10,
            metric="distortion",
            timings=True,
            locate_elbow=True,
            **kwargs
    ):
        super(KElbowVisualizer, self).__init__(estimator, ax=ax, **kwargs)

        # Get the scoring method
        if metric not in KELBOW_SCOREMAP:
            raise YellowbrickValueError(
                "'{}' is not a defined metric "
                "use one of distortion, silhouette, or calinski_harabasz"
            )

        # Store the arguments
        self.scoring_metric = KELBOW_SCOREMAP[metric]
        self.metric = metric
        self.timings = timings
        self.locate_elbow = locate_elbow

        # Convert K into a tuple argument if an integer
        if isinstance(k, int):
            self.k_values_ = list(range(2, k + 1))
        elif (
                isinstance(k, tuple)
                and len(k) == 2
                and all(isinstance(x, (int, np.integer)) for x in k)
        ):
            self.k_values_ = list(range(*k))
        elif isinstance(k, Iterable) and all(
                isinstance(x, (int, np.integer)) for x in k
        ):
            self.k_values_ = list(k)
        else:
            raise YellowbrickValueError(
                (
                    "Specify an iterable of integers, a range, or maximal K value,"
                    " the value '{}' is not a valid argument for K.".format(k)
                )
            )

        # Holds the values of the silhoutte scores
        self.k_scores_ = None
        # Set Default Elbow Value
        self.elbow_value_ = None

    def fit(self, X, y=None, **kwargs):
        """
        Fits n KMeans models where n is the length of ``self.k_values_``,
        storing the silhouette scores in the ``self.k_scores_`` attribute.
        The "elbow" and silhouette score corresponding to it are stored in
        ``self.elbow_value`` and ``self.elbow_score`` respectively.
        This method finishes up by calling draw to create the plot.
        """

        self.k_scores_ = []
        self.k_timers_ = []
        self.estimators_ = []

        for k in self.k_values_:
            # Compute the start time for each  model
            start = time.time()
            # Set the k value and fit the model
            estimator = clone(self.estimator)
            estimator.set_params(n_clusters=k)
            estimator.fit(X, **kwargs)
            # Append the time and score to our plottable metrics
            self.k_timers_.append(time.time() - start)
            # get X_hat_ return from fit if exist
            # avoid: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            self.k_scores_.append(
                self.scoring_metric(X if not hasattr(estimator, "X_hat_") else estimator.X_hat_, estimator.labels_))
            self.estimators_.append(estimator)

        if self.locate_elbow:
            locator_kwargs = {
                "distortion": {
                    "curve": "convex",
                    "direction": "decreasing",
                },
                "silhouette": {
                    "curve": "concave",
                    "direction": "increasing",
                },
                "calinski_harabasz": {
                    "curve": "concave",
                    "direction": "increasing",
                },
            }.get(self.metric, {})
            elbow_locator = KneeLocator(
                self.k_values_, self.k_scores_, **locator_kwargs
            )
            if elbow_locator.knee is None:
                self.elbow_value_ = None
                self.elbow_score_ = 0
                warning_message = (
                    "No 'knee' or 'elbow' point detected, "
                    "pass `locate_elbow=False` to remove the warning"
                )
                warnings.warn(warning_message, YellowbrickWarning)
            else:
                self.elbow_value_ = elbow_locator.knee
                self.elbow_score_ = self.k_scores_[self.k_values_.index(self.elbow_value_)]
                self.best_estimator_ = self.estimators_[self.k_values_.index(self.elbow_value_)]

        self.draw()

        return self

    def draw(self):
        """
        Draw the elbow curve for the specified scores and values of K.
        """
        # Plot the silhouette score against k
        self.ax.plot(self.k_values_, self.k_scores_, marker="D")
        if self.locate_elbow is True and self.elbow_value_ is not None:
            elbow_label = "elbow at $k={}$, $score={:0.3f}$".format(
                self.elbow_value_, self.elbow_score_
            )
            self.ax.axvline(
                self.elbow_value_, c=LINE_COLOR, linestyle="--", label=elbow_label
            )

        # If we're going to plot the timings, create a twinx axis
        if self.timings:
            self.axes = [self.ax, self.ax.twinx()]
            self.axes[1].plot(
                self.k_values_,
                self.k_timers_,
                label="fit time",
                c="g",
                marker="o",
                linestyle="--",
                alpha=0.75,
            )

        return self.ax

    def finalize(self):
        """
        Prepare the figure for rendering by setting the title as well as the
        X and Y axis labels and adding the legend.

        """
        # Get the metric name
        metric = self.scoring_metric.__name__.replace("_", " ").title()

        # Set the title
        self.set_title("{} Elbow for {} Clustering".format(metric, self.name))

        # Set the x and y labels
        self.ax.set_xlabel("k")
        self.ax.set_ylabel(metric.lower())

        # set the legend if locate_elbow=True
        if self.locate_elbow is True and self.elbow_value_ is not None:
            self.ax.legend(loc="best", fontsize="medium", frameon=True)

        # Set the second y axis labels
        if self.timings:
            self.axes[1].grid(False)
            self.axes[1].set_ylabel("fit time (seconds)", color="g")
            self.axes[1].tick_params("y", colors="g")


def optimal_clustering(X: Union[pd.DataFrame, np.ndarray],
                       size_min: int = 10,
                       metric: str = "distortion",
                       features_reduction: Optional[str] = None,
                       n_components: int = 2,
                       n_jobs: int = 1,
                       random_state: Optional[int] = None,
                       visualize: bool = False) \
        -> Tuple[KMeansConstrained, Union[umap.UMAP, PCA], int, np.ndarray, int, Sequence[
            np.float], np.float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to find the optimal clustering using a constrained k means. Two method are available to find the optimal
    number of cluster ``silhouette`` or ``elbow``.

    :param X: array-like or DataFrame of floats, shape (n_samples, n_features)
        The observations to cluster.
    :param size_min: Constrain the label assignment so that each cluster has a minimum size of size_min.
        If None, no constrains will be applied. default: None
    :param metric: Select the scoring metric to evaluate the clusters.
        The default is the mean distortion, defined by the sum of squared distances between each observation
        and its closest centroid. Other metrics include:
        - distortion: mean sum of squared distances to centers
        - silhouette: mean ratio of intra-cluster and nearest-cluster distance
        - calinski_harabasz: ratio of within to between cluster dispersion
    :param features_reduction: Method for reduction of the embedded space with n_components. Can be pca or umap.
        Default: None
    :param n_components: Dimension of the embedded space for features reduction. Default 2.
    :param n_jobs: int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    :param random_state: int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param visualize: bool, default: False
        If True, calls ``show()``
    :return:
        - cls:
          KMeansConstrained classifier object
        - cls_features_reduction:
          PCA or UMAP reduction object
        - cluster_nb:
          optimal number of cluster
        - labels:
          label[i] is the code or index of the centroid the i'th observation is closest to.
        - ref_label:
          cluster label with the minimal within-cluster sum-of-squares.
        - wicss_clusters:
          within-cluster sum-of-squares for each cluster
        - best_wicss_cluster:
          minimal wicss.
        - centroid:
          Centroids found at the last iteration of k-means.
        - X_hat:
          Copy of X with the missing values filled in.
    """
    assert metric in ["distortion", "silhouette", "calinski_harabasz"], "method need to be " \
                                                                        "'distortion' or " \
                                                                        "'silhouette' or " \
                                                                        "'calinski_harabasz'"

    n_samples = X.shape[0]
    min_cluster = 2 if metric in ["silhouette", "calinski_harabasz"] else 1
    max_cluster = n_samples // size_min + 1

    K = range(min_cluster, max_cluster)

    assert K, ValueError(f"Number of cluster is incompatible with metric {metric}, min cluster is {min_cluster} and "
                         f"max cluster is {max_cluster - 1} with a size_min of {size_min}")

    if max_cluster - 1 == 1:
        warnings.warn("Only one cluster is possible", RuntimeWarning, stacklevel=2)

    elbow_visualizer = KElbowVisualizer(estimator=KMeansConstrainedMissing(size_min=size_min,
                                                                           em_iter=10,
                                                                           features_reduction=features_reduction,
                                                                           n_components=n_components,
                                                                           n_jobs=n_jobs,
                                                                           random_state=random_state),
                                        k=K,
                                        metric=metric,
                                        locate_elbow=True)

    # fit
    elbow_visualizer.fit(X)

    if visualize:
        elbow_visualizer.show()

    cluster_nb = elbow_visualizer.elbow_value_

    # get cls, labels, centroids, inertia, Xhat corresponding to cluster_nb
    if cluster_nb == 1:
        warnings.warn("Only one cluster")
    if not cluster_nb:
        raise ValueError("Unable to determine a number of cluster")

    cls = elbow_visualizer.best_estimator_
    cls_features_reduction = elbow_visualizer.best_estimator_.cls_features_reduction_
    labels = elbow_visualizer.best_estimator_.labels_
    centroids = elbow_visualizer.best_estimator_.centroids_
    X_hat = elbow_visualizer.best_estimator_.X_hat_
    mu = elbow_visualizer.best_estimator_.mu_

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

    return cls, cls_features_reduction, cluster_nb, labels, ref_label, wicss_clusters, best_wicss_cluster, centroids, X_hat, mu
