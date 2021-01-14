from typing import Optional, Sequence, Tuple, Union, List
import warnings
import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from combat_harmonization.neurocombat import _check_exist_vars


# TODO: do combat by modality
# TODO: need to provide modality in df and other machine tag (ADD slice spacing)
# # https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb

def column_var_dtype(df: pd.DataFrame, identify_dtypes: Sequence[str] = ("object",)):
    """
    identify type of columns in DataFrame
    :param df: input dataframe
    :param identify_dtypes: pandas dtype, see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
    :return: summary df with col index and col name for all identify_dtypes vars
    """
    col_type = df.dtypes
    col_names = list(df)

    cat_var_index = [i for i, x in enumerate(col_type) if x in identify_dtypes]
    cat_var_name = [x for i, x in enumerate(col_names) if i in cat_var_index]

    cat_var_df = pd.DataFrame({"ind": cat_var_index,
                               "name": cat_var_name})

    return cat_var_df


def one_hot_encoder(df: pd.DataFrame, columns: List[str], drop_column: bool = True):
    """
    Encoding categorical feature in the dataframe, allow possibility to keep NaN.
    The categorical feature index and name are from cat_var function. These columns need to be "object" dtypes.
    :param df: input dataframe
    :param columns: List of columns to encode
    :param drop_column: Set to True to drop the original column after encoding. Default to True.
    :return:
        df: new dataframe where columns are one hot encoded
    """

    _check_exist_vars(df, columns)

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        if drop_column:
            df = df.drop(col, axis=1)

    return df


def scaler_encoder(df: pd.DataFrame, columns: List[str], scaler=StandardScaler()):
    """
    Apply sklearn scaler to columns.
    :param df: input dataframe
    :param columns: List of columns to encode
    :param scaler: scaler object from sklearn
    :return df: new dataframe where columns are scaler encoded
    """

    le = scaler
    df[columns] = le.fit_transform(df[columns])

    return df


def kmeans_constrained_missing(X: Union[pd.DataFrame, np.ndarray], n_clusters: int, size_min: Optional[int] = None,
                               max_iter: int = 10, random_state: Optional[int] = None) \
        -> Tuple[KMeansConstrained, np.ndarray, np.ndarray, np.float, np.ndarray]:
    """
    K-Means combat_harmonization with minimum and maximum cluster size constraints with the possibility of missing values.
    # inspired of https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data
    :param X: array-like or DataFrame of floats, shape (n_samples, n_features)
        The observations to cluster.
    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :param size_min: Constrain the label assignment so that each cluster has a minimum size of size_min.
        If None, no constrains will be applied. default: None
    :param max_iter: Maximum number of EM iterations to perform. default: 10
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

        # perform combat_harmonization on the filled-in data
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
                       random_state: Optional[int] = None) \
        -> Tuple[KMeansConstrained, int, np.ndarray, int, Sequence[np.float], np.float, np.ndarray, np.ndarray]:
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
        cls, labels, centroids, inertia, X_hat = kmeans_constrained_missing(X, n_clusters=k, size_min=size_min,
                                                                            max_iter=10,
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
        print(sil)
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
    D = cdist(X, centroids, 'euclidean')
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





from matplotlib import pyplot as plt

# X, true_labels, Xm = make_fake_data(fraction_missing=0.3, n_clusters=5, seed=0)
# labels, centroids, inertia, X_hat = kmeans_constrained_missing(Xm, n_clusters=2)

# # plot the inferred points, color-coded according to the true cluster labels
# fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d', 'aspect': 'auto'})
# ax[0].scatter3D(X[:, 0], X[:, 1], X[:, 2], c=true_labels, )  #   cmap='gist_rainbow'
# ax[1].scatter3D(X_hat[:, 0], X_hat[:, 1], X_hat[:, 2], c=true_labels)
# ax[0].set_title('Original data')
# ax[1].set_title('Imputed (30% missing values)')
# fig.tight_layout()
#
# true_labels = df['dicom_file'].apply(lambda x: 0 if "1-5T" in x else 1).to_list()
# true_labels = df['MagneticFieldStrength'].to_list()
# true_labels = [j for i, j in enumerate(true_labels) if i % 2 == 0]
# true_labels = [1 if x == 1.5 else 0 for x in true_labels]  #
#
# final_df = final_df.drop(["AccessionNumber", "InstitutionName_0", ""], axis=1)
# d = final_df[["MagneticFieldStrength", "ManufacturerModelName_0", "StationName_0"]]
# Xm = final_df.to_numpy()
#
# import pandas as pd
#
# df = pd.read_csv(
#     "/media/acarre/Data/PythonProjects/these_all/ComBat/data_processing/output_dicom_tag/SainteAnne_metadata1.csv")
# df = df.sample(frac=1, axis=0).reset_index(drop=True)
#
# # df = df[df["dicom_file"].str.contains("flair")]
# df = df.drop('dicom_file', axis=1)
# df_split = df
#
# dtype_df = df.select_dtypes(exclude=["int64", "float64"])
# df_split = df_split.drop(dtype_df.columns.to_list(), axis=1)
# dtype_df = dtype_df.applymap(lambda x: x.strip("][").split(", ") if not isinstance(x, (float, int)) else x)
#
# final_exploded_df = pd.DataFrame()
# for column in dtype_df:
#     df_exploded = pd.DataFrame(dtype_df[column].explode())
#     df_exploded["index"] = df_exploded.groupby(df_exploded.index).cumcount()  # Pivot and remove the column axis name.
#     df_exploded["index"] = df_exploded["index"].apply(lambda x: column + "_" + str(x))
#     df_exploded = df_exploded.pivot(columns="index", values=column).rename_axis(None, axis=1)
#     print(df_exploded)
#     final_exploded_df = pd.concat([final_exploded_df, df_exploded], axis=1)
#     # final_exploded_df.join(df_exploded)
# final_df = pd.concat([df_split, final_exploded_df], axis=1)
# final_df = final_df.applymap(lambda x: pd.to_numeric(x, errors="ignore"))