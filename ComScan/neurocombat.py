# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
Created on: Jan 14, 2021
"""
import os
import pickle
import warnings
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd
from neuroCombat.neuroCombat import make_design_matrix, standardize_across_features, fit_LS_model_and_find_priors, \
    find_parametric_adjustments, find_non_parametric_adjustments, find_non_eb_adjustments
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from ComScan.clustering import optimal_clustering
from ComScan.nifti import _compute_mask_files, flatten_nifti_files
from ComScan.utils import get_column_index, one_hot_encoder, scaler_encoder, check_is_nii_exist, \
    load_nifty_volume_as_array, save_to_nii


def _check_exist_vars(df: pd.DataFrame, _vars: List) -> np.ndarray:
    """
    Check that a list of columns name exist in a DataFrame.
    :param df: a DataFrame
    :param _vars: List of columns name to check
    :return index of columns name
    :raise Value error if missing features
    """
    column_index = get_column_index(df, _vars)
    is_feature_present = column_index != -1
    if not isinstance(_vars, np.ndarray):
        _vars = np.array(_vars)
    if not is_feature_present.all():
        raise ValueError(f"Missing features: {', '.join(_vars[~is_feature_present].astype(str))}")
    return column_index


def _check_single_covariate_sample(df: pd.DataFrame, _vars: List) -> None:
    """
    Check if samples present single covariate
    :param df: a DataFrame
    :param _vars: List of columns name to check
    :raise ValueError if a covariate contain a unique sample
    """
    for _var in _vars:
        if 1 in df[_var].value_counts().tolist():
            raise ValueError(f"Combat ComScan requires more than one sample. "
                             f"The following covariate contain a unique sample: {_var} ")


def _check_nans(df: pd.DataFrame) -> None:
    """
    Check if NaNs are present in dataframe
    :param df: a DataFrame
    :raise: ValueError if NaNs are present
    """
    if df.isnull().values.any():
        raise ValueError("NaN values found on data. \n"
                         "Combat can not work with NaN values, maybe drop samples or features columns"
                         " containing these values")


class Combat(BaseEstimator, TransformerMixin):
    """
    Harmonize/normalize features using Combat's [1] parametric empirical Bayes framework

    [1] Fortin, Jean-Philippe, et al. "Harmonization of cortical thickness
    measurements across scanners and sites." Neuroimage 167 (2018): 104-1

    Parameters
    ----------
    features : Target features to be harmonized

    sites : Target variable for ComScan problems (e.g. acquisition sites or scanner).

    discrete_covariates : Target covariates which are categorical (e.g. male or female).

    continuous_covariates : Target covariates which are continuous (e.g. age).

    ref_site : Variable value (acquisition sites or scanner) to be used as reference for batch adjustment.
        Default is False.

    empirical_bayes : Performed empirical bayes.
        Default is True.

    parametric : Performed parametric adjustements.
        Default is True.

    mean_only : Adjust only the mean (no scaling)
        Default is False.

    copy : Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).
        Default is True.

    Attributes
    ----------
    info_dict_fit_ : dictionary that stores batch info of fitted data with:
         batch_levels, ref_level, n_batch, n_sample, sample_per_batch, batch_info

    stand_mean_ : array-like
        Standardized mean

    var_pooled_ : array-like
         Variance pooled

    gamma_star_ : array-like
        Adjustement gamma star

    delta_star_ : array-like
        Adjustement delta star

    info_dict_transform_ : dictionary that stores batch info of transformed data with:
         batch_levels, ref_level, n_batch, n_sample, sample_per_batch, batch_info

    Examples
    --------
    >>> data = pd.DataFrame([{"features_1": 0.97, "features_2": 2, "sites": 0},
    >>> {"features_1": 1.35, "features_2": 1.01, "sites": 1}, {"features_1": 1.43, "features_2": 1.09, "sites": 1},
    >>> {"features_1": 0.85, "features_2": 2.3, "sites": 0}])

    >>> combat = Combat(features=["features_1", "features_2"], sites=["sites"], ref_site=1)
    >>> print(combat.fit(data))
    Combat(continuous_covariates=[], discrete_covariates=[],
       features=['features_1', 'features_2'], ref_site=1, sites=['sites'])
    >>> print(combat.gamma_star_)
    [[-11.85476756  27.30493785]
    [  0.           0.        ]]
    >>> print(combat.transform(data))
    [[1.40593957 1.01395564 0.        ]
    [1.35       1.01       1.        ]
    [1.43       1.09       1.        ]
    [1.37064296 1.08999992 0.        ]]

    Notes
    -----
    NaNs values are not treated.

    """

    def __init__(self,
                 features: Union[List[str], List[int], str, int],
                 sites: Union[str, int],
                 discrete_covariates: Optional[Union[List[str], List[int], str, int]] = None,
                 continuous_covariates: Optional[Union[List[str], List[int], str, int]] = None,
                 ref_site: Optional[Union[str, int]] = None,
                 empirical_bayes: bool = True,
                 parametric: bool = True,
                 mean_only: bool = False,
                 copy: bool = True) -> None:

        self.features = features
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.sites = sites
        self.ref_site = ref_site
        self.empirical_bayes = empirical_bayes
        self.parametric = parametric
        self.mean_only = mean_only
        self.copy = copy

    def __reset(self) -> None:
        """
        Reset internal data-dependent state of the combat fit, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'info_dict_fit_'):
            del self.info_dict_fit_
            del self.stand_mean_
            del self.var_pooled_
            del self.gamma_star_
            del self.delta_star_

    def fit(self, X: Union[np.ndarray, pd.DataFrame], *y: Optional[Union[np.ndarray, pd.DataFrame]]) -> "Combat":
        """
        Compute the stand mean, var pooled, gamma star, delta star to be used for later adjusted data.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features). Requires the columns needed by the Combat().
            The data used to find adjustments.
        *y : y in scikit learn: None
            Ignored.

        Returns
        -------
        self : object
            Fitted combat estimator.
        """
        self.__reset()

        columns_features, columns_discrete_covariates, columns_continuous_covariates, columns_sites, other_columns \
            = self._check_data(X)

        X = self._validate_data(X, copy=self.copy, estimator=self)

        if self.ref_site is None:
            ref_level = None
        else:
            ref_indices = np.argwhere(X[:, columns_sites[0]] == self.ref_site).squeeze()
            if ref_indices.shape[0] == 0:
                raise ValueError(f"ref_site: {self.ref_site} not found")
            else:
                ref_level = np.int(X[ref_indices[0], columns_sites])

        # create dictionary that stores batch info
        (batch_levels, sample_per_batch) = np.unique(X[:, columns_sites], return_counts=True)
        self.info_dict_fit_ = {
            'batch_levels': batch_levels,
            'ref_level': ref_level,
            'n_batch': len(batch_levels),
            'n_sample': int(X.shape[0]),
            'sample_per_batch': sample_per_batch.astype('int'),
            'batch_info': [list(np.where(X[:, columns_sites] == idx)[0]) for idx in batch_levels]
        }

        # create design matrix
        design = make_design_matrix(Y=X, batch_col=columns_sites,
                                    cat_cols=columns_discrete_covariates, num_cols=columns_continuous_covariates,
                                    ref_level=ref_level)

        # standardize data across features
        s_data, self.stand_mean_, self.var_pooled_ = standardize_across_features(X=X[:, columns_features].T,
                                                                                 design=design,
                                                                                 info_dict=self.info_dict_fit_)

        # fit L/S models and find priors
        LS_dict = fit_LS_model_and_find_priors(s_data=s_data, design=design,
                                               info_dict=self.info_dict_fit_,
                                               mean_only=self.mean_only)

        # find parametric adjustments
        if self.empirical_bayes:
            if self.parametric:
                self.gamma_star_, self.delta_star_ = find_parametric_adjustments(s_data=s_data,
                                                                                 LS=LS_dict,
                                                                                 info_dict=self.info_dict_fit_,
                                                                                 mean_only=self.mean_only)
            else:
                self.gamma_star_, self.delta_star_ = find_non_parametric_adjustments(s_data=s_data,
                                                                                     LS=LS_dict,
                                                                                     info_dict=self.info_dict_fit_,
                                                                                     mean_only=self.mean_only)
        else:
            self.gamma_star_, self.delta_star_ = find_non_eb_adjustments(s_data=s_data, LS=LS_dict,
                                                                         info_dict=self.info_dict_fit_)

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Scale features of X according to combat estimator.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features). Requires the columns needed by the Combat().
            Input data that will be transformed.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        columns_features, columns_discrete_covariates, columns_continuous_covariates, columns_sites, other_columns \
            = self._check_data(X, check_single_covariate=False)

        X = self._validate_data(X, copy=self.copy, estimator=self)

        (batch_levels, sample_per_batch) = np.unique(X[:, columns_sites], return_counts=True)

        # Check all sites from new_data were seen
        if not all(site_name in self.info_dict_fit_["batch_levels"] for site_name in batch_levels):
            raise ValueError(f"There is an unseen site during the fit method in the data")

        # create dictionary that stores batch info
        self.info_dict_transform_ = {
            'batch_levels': batch_levels,
            'ref_level': self.ref_site,
            'n_batch': len(batch_levels),
            'n_sample': int(X.shape[0]),
            'sample_per_batch': sample_per_batch.astype('int'),
            'batch_info': [list(np.where(X[:, columns_sites] == idx)[0]) for idx in batch_levels]
        }
        # create design matrix
        design = make_design_matrix(Y=X, batch_col=columns_sites,
                                    cat_cols=columns_discrete_covariates, num_cols=columns_continuous_covariates,
                                    ref_level=self.info_dict_transform_["ref_level"])

        # create design to take into account: One transform or Missing sites compare to fit
        design_batch = np.eye(self.info_dict_fit_["n_batch"])[X[:, columns_sites[0]].astype(int)]
        design = np.concatenate((design_batch, design[:, self.info_dict_transform_["n_batch"]:]), axis=1)

        self.stand_mean_transform_ = np.delete(self.stand_mean_,
                                               range(self.stand_mean_.shape[1] - self.info_dict_transform_[
                                                   "n_sample"]), axis=1)

        s_data = ((X[:, columns_features].T - self.stand_mean_transform_) / np.dot(np.sqrt(self.var_pooled_),
                                                                                   np.ones(
                                                                                       (1, self.info_dict_transform_[
                                                                                           "n_sample"]))))
        # adjust data
        # ** obligatory to match the fit data for adjust_data (sample_per_batch, n_sample, batch_info)
        bayes_data = self._adjust_final_data(s_data=s_data, design=design, sample_per_batch=np.array(
            [np.count_nonzero(X[:, columns_sites] == lvl) for lvl in self.info_dict_fit_["batch_levels"]]),
                                             n_sample=self.info_dict_transform_["n_sample"],
                                             batch_info=[list(np.where(X[:, columns_sites] == idx)[0]) for idx in
                                                         self.info_dict_fit_["batch_levels"]]).T

        X[:, columns_features] = bayes_data

        return X

    def _check_data(self, X: Union[np.ndarray, pd.DataFrame], check_single_covariate: bool = True) -> Tuple[List, List,
                                                                                                            List, List,
                                                                                                            List]:
        """
        Check that the input data array-like or DataFrame of shape (n_samples, n_features) have all the required
        format needed by the Combat()
        :param X: input data array-like or DataFrame of shape (n_samples, n_features)
        :param check_single_covariate: check single covariate
        :return: idx of: columns_features, columns_discrete_covariates,
                         columns_continuous_covariates, columns_sites, other_columns
        """
        self.features, self.discrete_covariates, self.continuous_covariates, self.sites = map(
            lambda x: [x] if isinstance(x, (str, int)) else x if x is not None else [],
            [self.features, self.discrete_covariates, self.continuous_covariates, self.sites])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.copy())

        columns_features = _check_exist_vars(X, self.features)
        columns_sites = _check_exist_vars(X, self.sites)
        columns_used = (columns_features, columns_sites)
        columns_discrete_covariates, columns_continuous_covariates = [], []
        if self.discrete_covariates:
            columns_discrete_covariates = _check_exist_vars(X, self.discrete_covariates)
            _check_single_covariate_sample(X, self.discrete_covariates)
            columns_used += (columns_discrete_covariates,)
        if self.continuous_covariates:
            columns_continuous_covariates = _check_exist_vars(X, self.continuous_covariates)
            _check_single_covariate_sample(X, self.continuous_covariates)
            columns_used += (columns_continuous_covariates,)

        unq, unq_cnt = np.unique(np.concatenate((X.columns.get_indexer(X.columns), np.concatenate(columns_used))),
                                 return_counts=True)
        other_columns = unq[unq_cnt == 1]

        if check_single_covariate:
            _check_single_covariate_sample(X, self.sites)
            if self.discrete_covariates:
                _check_single_covariate_sample(X, self.discrete_covariates)
            if self.continuous_covariates:
                _check_single_covariate_sample(X, self.continuous_covariates)

        columns_features, columns_discrete_covariates, columns_continuous_covariates, columns_sites, other_columns \
            = map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                  [columns_features, columns_discrete_covariates,
                   columns_continuous_covariates, columns_sites,
                   other_columns])

        _check_nans(X[self.features + self.discrete_covariates + self.continuous_covariates + self.sites])

        return columns_features, columns_discrete_covariates, columns_continuous_covariates, columns_sites, other_columns

    def _adjust_final_data(self, s_data, design, sample_per_batch, n_sample, batch_info):
        """
        Adjust final data
        :param s_data: array-like standardized data
        :param design: array-like design matrix
        :param sample_per_batch: number of sample per batch
        :param n_sample: total number of sample
        :param batch_info: batch info is batch index for n_batch
        :return:
        """
        bayes_data = s_data
        batch_design = design[:, :self.info_dict_fit_["n_batch"]]

        for j, batch_idxs in enumerate(batch_info):
            if not batch_idxs:
                continue
            dsq = np.sqrt(self.delta_star_[j, :])
            dsq = dsq.reshape((len(dsq), 1))
            denom = np.dot(dsq, np.ones((1, sample_per_batch[j])))
            numer = np.array(bayes_data[:, batch_idxs] - np.dot(batch_design[batch_idxs, :], self.gamma_star_).T)

            bayes_data[:, batch_idxs] = numer / denom

        vpsq = np.sqrt(self.var_pooled_).reshape((len(self.var_pooled_), 1))
        bayes_data = bayes_data * np.dot(vpsq, np.ones((1, n_sample))) + self.stand_mean_transform_

        return bayes_data

    def save_fit(self, filepath: str) -> None:
        """
        save a fitted model attribute (info_dict_fit_, stand_mean_, var_pooled_, gamma_star_, delta_star_)
        :param filepath: filepath were to save. if no extension .pkl will add it
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)

        if not filepath.endswith(".pkl"):
            filepath += ".pkl"

        attrs_to_save = {k: v for k, v in vars(self).items()
                         if k.endswith("_") and not k.startswith("__")}

        with open(filepath, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(attrs_to_save, output, pickle.HIGHEST_PROTOCOL)

    def load_fit(self, filepath: str) -> None:
        """
        load a fitted model attribute (info_dict_fit_, stand_mean_, var_pooled_, gamma_star_, delta_star_)
        :param filepath: filepath of the pkl file to load
        """
        with open(filepath, 'rb') as pickle_file:  # Overwrites any existing file.
            loaded_pickle = pickle.load(pickle_file)
        for k, v in loaded_pickle.items():
            setattr(self, k, v)


class AutoCombat(Combat):
    """
    Harmonize/normalize features using Combat's [1] parametric empirical Bayes framework.

    Combat need to have well-known acquisition sites or scanner to harmonize features.
    It is sometimes difficult to define an imaging acquisition site if on two sites imaging parameters
    can be really similar. ComScan gives the possibility to automatically determine the number of sites
    and their association based on imaging features (e.g. dicom tags) by clustering.
    Thus ComScan can be used on data not seen during training because it can predict which imager best matches
    the one it has seen during training.


    [1] Fortin, Jean-Philippe, et al. "Harmonization of cortical thickness
    measurements across scanners and sites." Neuroimage 167 (2018): 104-1

    Parameters
    ----------
    features : Target features to be harmonized.

    sites_features : Target variable for define (acquisition sites or scanner) by clustering.

    sites: Target variable for ComScan problems (e.g. acquisition sites or scanner).
           This argument is Optional. If this argument is provided will run traditional ComBat.
           In this case args: sites_features, size_min, method, scaler_clustering, discrete_cluster_features,
            continuous_cluster_features, threshold_missing_sites_features, drop_site_columns
           are unused.

    size_min: Constraint of the minimum size of site for clustering.

    method: "silhouette" or "elbow". Method to define the optimal number of cluster. Default: silhouette.

    use_ref_site: Use a ref site to be used as reference for batch adjustment. The ref site used is the cluster
     with the minimal inertia. i.e minimizing within-cluster sum-of-squares.

    scaler_clustering: Scaler to use for continuous site features. Need to be a scikit learn scaler.
        Default: StandardScaler()

    discrete_cluster_features: Target sites_features which are categorical to one-hot (e.g. ManufacturerModelName).

    continuous_cluster_features: Target sites_features which are continuous to scale (e.g. EchoTime).

    threshold_missing_sites_features: Threshold of acceptable missing features for sites features clustering.
        25 specify that 75% of all samples need to have this features. Default 25.

    drop_site_columns: Drop sites columns find by clustering in return.

    discrete_combat_covariates : Target covariates which are categorical (e.g. male or female).

    continuous_combat_covariates : Target covariates which are continuous (e.g. age).

    empirical_bayes : Performed empirical bayes.
        Default is True.

    parametric : Performed parametric adjustements.
        Default is True.

    mean_only : Adjust only the mean (no scaling)
        Default is False.

    random_state: int, RandomState instance or None, optional, default: 123
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy : Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).
        Default is True.

    Attributes
    ----------

    cls_ : clustering classifier object

    info_clustering_ : Dictionary that stores info of clustering from sites_features with cluster_nb, labels, ref_label
        wicss_clusters, best_wicss_cluster

    Examples
    --------
    >>> data = pd.DataFrame([{"features_1": 0.97, "site_features_0": 2, "site_features_1": 0},
    >>> {"features_1": 1.35, "site_features_0": 1.01, "site_features_1": 1},
    >>> {"features_1": 1.43, "site_features_0": 1.09, "site_features_1": 1},
    >>> {"features_1": 0.85, "site_features_0": 2.3, "site_features_1": 0}])

    >>> auto_combat = AutoCombat(features=["features_1"], sites_features=["site_features_0", "site_features_1"],
    >>> continuous_cluster_features=["site_features_0", "site_features_1"])
    >>> print(auto_combat.fit(data))
    AutoCombat(continuous_cluster_features=['site_features_0', 'site_features_1'],
           discrete_cluster_features=[], features=['features_1'],
           sites=['sites'],
           sites_features=['site_features_0', 'site_features_1'], size_min=2)
    Notes
    -----
    NaNs values are not treated.

    """

    def __init__(self,
                 features: Union[List[str], List[int], str, int],
                 sites_features: Union[List[str], List[int], str, int] = None,
                 sites: Optional[Union[str, int]] = None,
                 size_min: int = 10,
                 method: str = "silhouette",
                 use_ref_site: bool = False,
                 scaler_clustering=StandardScaler(),
                 discrete_cluster_features: Optional[Union[List[str], List[int], str, int]] = None,
                 continuous_cluster_features: Optional[Union[List[str], List[int], str, int]] = None,
                 threshold_missing_sites_features=25,
                 drop_site_columns: bool = True,
                 discrete_combat_covariates: Optional[Union[List[str], List[int], str, int]] = None,
                 continuous_combat_covariates: Optional[Union[List[str], List[int], str, int]] = None,
                 empirical_bayes: bool = True,
                 parametric: bool = True,
                 mean_only: bool = False,
                 random_state: Union[int, None] = 123,
                 copy: bool = True) -> None:
        super().__init__(features=features,
                         sites=sites,
                         discrete_covariates=discrete_combat_covariates,
                         continuous_covariates=continuous_combat_covariates,
                         empirical_bayes=empirical_bayes,
                         parametric=parametric,
                         mean_only=mean_only)

        if not 0 < threshold_missing_sites_features < 100:
            raise ValueError("threshold_missing_sites_features need to be comprise between 0 and 100")

        if (sites_features is None and sites is None) or (sites_features is not None and sites is not None):
            raise ValueError("one of sites_features or sites must be provided. If sites will run traditional ComBat")

        self.features = features
        self.sites_features = sites_features
        self.size_min = size_min
        self.method = method
        self.use_ref_site = use_ref_site
        self.scaler_clustering = scaler_clustering
        self.discrete_cluster_features = discrete_cluster_features
        self.continuous_cluster_features = continuous_cluster_features
        self.threshold_missing_sites_features = threshold_missing_sites_features
        self.drop_site_columns = drop_site_columns
        self.discrete_combat_covariates = discrete_combat_covariates
        self.continuous_combat_covariates = continuous_combat_covariates
        self.empirical_bayes = empirical_bayes
        self.parametric = parametric
        self.mean_only = mean_only
        self.random_state = random_state
        self.copy = copy

    def __reset(self) -> None:
        """
        Reset internal data-dependent state of the AutoCombat fit, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'cls_', ):
            del self.cls_
            del self.info_clustering_

    def fit(self, X: Union[np.ndarray, pd.DataFrame], *y: Optional[Union[np.ndarray, pd.DataFrame]]) -> "AutoCombat":
        """
        Compute sites, ref_site using clustering. Then compute the stand mean, var pooled, gamma star, delta star
        to be used for later adjusted data from Combat.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features).
            Requires the columns needed by the ComScan().
            The data used to find adjustments.
        *y : y in scikit learn: None
            Ignored.

        Returns
        -------
        self : object
            Fitted ComScan estimator.
        """
        self.__reset()

        if self.sites_features is not None:

            clustering_data, columns_clustering_features, columns_discrete_cluster_features, \
                columns_continuous_cluster_features = self._check_data_cluster(X)

            clustering_data = self._validate_data(clustering_data, copy=self.copy, estimator=self)

            self.cls_, cluster_nb, labels, ref_label, wicss_clusters, best_wicss_cluster, _, _ = optimal_clustering(
                X=clustering_data,
                size_min=self.size_min,
                method=self.method)

            self.info_clustering_ = {
                'cluster_nb': cluster_nb,
                'labels': labels,
                'ref_label': ref_label,
                'wicss_clusters': wicss_clusters,
                'best_wicss_cluster': best_wicss_cluster,
            }

            if cluster_nb == 1:
                raise ValueError("Combat can not run. "
                                 "Only one acquisition site found from sites features. "
                                 "Are the data really different or from different acquisition site?")

            # add sites columns
            X = self._add_sites(X, labels)

            if self.use_ref_site:
                self.ref_site = ref_label

        super().fit(X, *y)

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Scale features of X according to combat estimator.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features). Requires the columns needed by the Combat().
            Input data that will be transformed.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        if self.sites_features is not None:
            clustering_data, columns_clustering_features, columns_discrete_cluster_features, \
            columns_continuous_cluster_features = self._check_data_cluster(X)

            clustering_data = self._validate_data(clustering_data, copy=self.copy, estimator=self)

            # get labels for sites
            labels = self.cls_.predict(clustering_data)

            # add sites columns
            X = self._add_sites(X, labels)

        X = super().transform(X)

        if self.sites_features is not None:
            # drop added sites columns
            if self.drop_site_columns:
                X = X[:, :-1]

        return X

    def _check_data_cluster(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, List, List, List]:
        """
        Check that the input data array-like or DataFrame of shape (n_samples, n_features) have all the required
        format needed by the Combat()
        :param X: input data array-like or DataFrame of shape (n_samples, n_features)
        :return: idx of: columns_clustering_features
        """

        if not self.sites_features:
            raise ValueError("sites_features is empty")

        self.sites_features, self.discrete_cluster_features, self.continuous_cluster_features = map(
            lambda x: [x] if isinstance(x, (str, int)) else x if x is not None else [],
            [self.sites_features, self.discrete_cluster_features, self.continuous_cluster_features])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.copy())

        columns_clustering_features = _check_exist_vars(X, self.sites_features)

        columns_used = (columns_clustering_features,)
        columns_discrete_cluster_features, columns_continuous_cluster_features = [], []
        if self.discrete_cluster_features:
            columns_discrete_cluster_features = _check_exist_vars(X, self.discrete_cluster_features)
            columns_used += (columns_discrete_cluster_features,)
        if self.continuous_cluster_features:
            columns_continuous_cluster_features = _check_exist_vars(X, self.continuous_cluster_features)
            columns_used += (columns_continuous_cluster_features,)

        unq, unq_cnt = np.unique(np.concatenate((columns_clustering_features, np.concatenate(columns_used))),
                                 return_counts=True)
        other_columns = unq[unq_cnt == 1]

        if other_columns.size > 0:
            warnings.warn(
                f"Some columns: {X.columns[other_columns].tolist()} are not specified as discrete or continuous."
                f"Clustering will interpret this data as raw.")

        percent_missing = X.iloc[:, columns_clustering_features].isnull().sum() * 100 / len(X)
        percent_missing = percent_missing.to_dict()

        for features, val in percent_missing.items():
            if val > (100 - self.threshold_missing_sites_features):
                warnings.warn(f"sites_features: {features} removed because more than "
                              f"{self.threshold_missing_sites_features}% of missing data")
                columns_clustering_features = np.delete(columns_clustering_features,
                                                        np.where(columns_clustering_features ==
                                                                 get_column_index(X, [features])[0]))

        columns_clustering_features, columns_discrete_cluster_features, columns_continuous_cluster_features = map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
            [columns_clustering_features, columns_discrete_cluster_features, columns_continuous_cluster_features])

        clustering_data = self._validate_data(X, copy=self.copy, estimator=self)

        if columns_discrete_cluster_features:
            clustering_data = one_hot_encoder(df=pd.DataFrame(clustering_data),
                                              columns=columns_discrete_cluster_features)

        if columns_continuous_cluster_features:
            clustering_data = scaler_encoder(df=pd.DataFrame(clustering_data),
                                             columns=columns_continuous_cluster_features,
                                             scaler=self.scaler_clustering)

        clustering_data = clustering_data.iloc[:, columns_clustering_features]

        return clustering_data, columns_clustering_features, columns_discrete_cluster_features, columns_continuous_cluster_features

    def _add_sites(self, X: Union[np.ndarray, pd.DataFrame], labels: Union[np.ndarray, List]) \
            -> Union[np.ndarray, pd.DataFrame]:
        """
        Add sites find by clustering to X
        :param X: input data array-like or DataFrame of shape (n_samples, n_features)
        :param labels: labels of sites find by clustering
        """

        if self.copy:
            X = X.copy(deep=True)

        sites = None
        if isinstance(X, pd.DataFrame):
            X['sites'] = labels
            sites = 'sites'
        elif isinstance(X, np.ndarray):
            X = np.c_[X, labels]
            sites = X.shape[1]

        self.sites = sites

        return X


class ImageCombat(AutoCombat):
    """
    Harmonize/normalize features using Combat's [1] parametric empirical Bayes framework directly on image.

    ImageCombat allow the possibility to Harmonize/normalize a set of NIFTI images.
    All images must have the same dimensions and orientation. A common mask is created based on an heuristic
    proposed by T.Nichols. Images are then vectorizing for ComScan.
    ImageCombat allows the possibily to use Combat (well-defined site) or AutoCombat (clustering for sites finding)

    [1] Fortin, Jean-Philippe, et al. "Harmonization of cortical thickness
    measurements across scanners and sites." Neuroimage 167 (2018): 104-1

    Parameters
    ----------
    image_path: image_path of nifti files.

    sites_features : Target variable for define (acquisition sites or scanner) by clustering.

    sites: Target variable for ComScan problems (e.g. acquisition sites or scanner).
           This argument is Optional. If this argument is provided will run traditional ComBat.
           In this case args: sites_features, size_min, method, scaler_clustering, discrete_cluster_features,
            continuous_cluster_features, threshold_missing_sites_features, drop_site_columns
           are unused.

    size_min: Constraint of the minimum size of site for clustering.

    method: "silhouette" or "elbow". Method to define the optimal number of cluster. Default: silhouette.

    use_ref_site: Use a ref site to be used as reference for batch adjustment. The ref site used is the cluster
     with the minimal inertia. i.e minimizing within-cluster sum-of-squares.

    scaler_clustering: Scaler to use for continuous site features. Need to be a scikit learn scaler.
        Default: StandardScaler()

    discrete_cluster_features: Target sites_features which are categorical to one-hot (e.g. ManufacturerModelName).

    continuous_cluster_features: Target sites_features which are continuous to scale (e.g. EchoTime).

    threshold_missing_sites_features: Threshold of acceptable missing features for sites features clustering.
        25 specify that 75% of all samples need to have this features. Default 25.

    drop_site_columns: Drop sites columns find by clustering in return.

    discrete_combat_covariates : Target covariates which are categorical (e.g. male or female).

    continuous_combat_covariates : Target covariates which are continuous (e.g. age).

    empirical_bayes : Performed empirical bayes.
        Default is True.

    parametric : Performed parametric adjustements.
        Default is True.

    mean_only : Adjust only the mean (no scaling)
        Default is False.

    random_state: int, RandomState instance or None, optional, default: 123
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy : Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).
        Default is True.

    Attributes
    ----------

    mask_ : array-like of the common brain mask

    flattened_array_ : flattened array of all the training set


    Notes
    -----
    NaNs values are not treated.

    """

    def __init__(self, image_path: Union[str, int],
                 sites_features: Union[List[str], List[int], str, int] = None,
                 sites: Union[str, int] = None,
                 save_path_fit: str = 'fit_data',
                 save_path_transform: str = 'transform_data',
                 size_min: int = 10, method: str = "silhouette",
                 use_ref_site: bool = False,
                 scaler_clustering=StandardScaler(),
                 discrete_cluster_features: Optional[Union[List[str], List[int], str, int]] = None,
                 continuous_cluster_features: Optional[Union[List[str], List[int], str, int]] = None,
                 threshold_missing_sites_features=25,
                 drop_site_columns: bool = True,
                 discrete_combat_covariates: Optional[Union[List[str], List[int], str, int]] = None,
                 continuous_combat_covariates: Optional[Union[List[str], List[int], str, int]] = None,
                 empirical_bayes: bool = True,
                 parametric: bool = True, mean_only: bool = False,
                 random_state: Union[int, None] = 123,
                 flattened_dtype: Optional[np.dtype] = np.float16,
                 output_dtype: Optional[np.dtype] = np.float32,
                 copy: bool = True) -> None:
        super().__init__(features=[],
                         sites_features=sites_features,
                         sites=sites,
                         size_min=size_min,
                         method=method,
                         use_ref_site=use_ref_site,
                         scaler_clustering=scaler_clustering,
                         discrete_cluster_features=discrete_cluster_features,
                         continuous_cluster_features=continuous_cluster_features,
                         threshold_missing_sites_features=threshold_missing_sites_features,
                         drop_site_columns=drop_site_columns,
                         discrete_combat_covariates=discrete_combat_covariates,
                         continuous_combat_covariates=continuous_combat_covariates,
                         empirical_bayes=empirical_bayes,
                         parametric=parametric,
                         mean_only=mean_only,
                         random_state=random_state,
                         copy=copy)

        self.image_path = image_path
        self.save_path_fit = save_path_fit
        self.save_path_transform = save_path_transform
        self.flattened_dtype = flattened_dtype
        self.output_dtype = output_dtype

        # For sequential transform
        self.sequential_transform = True

    def __reset(self) -> None:
        """
        Reset internal data-dependent state of the ImageCombat fit, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'mask_', ):
            del self.mask_
            del self.flattened_array_

    def fit(self, X: Union[np.ndarray, pd.DataFrame], *y: Optional[Union[np.ndarray, pd.DataFrame]]) -> "ImageCombat":

        _, list_image_path = self._check_image_path(X)

        # create common brain mask
        self.mask_, _ = _compute_mask_files(input_path=list_image_path,
                                            output_path=os.path.join(self.save_path_fit, "common_mask.nii.gz"),
                                            return_mean=False)

        # flatten nifti files
        self.flattened_array_ = flatten_nifti_files(input_path=list_image_path, mask=self.mask_,
                                                    output_flattened_array_path=os.path.join(self.save_path_fit,
                                                                                             "flattened_array"),
                                                    dtype=self.flattened_dtype, save=True, compress_save=True)

        X, _ = self._add_voxels_as_features(X.copy(deep=True), self.flattened_array_)

        # run AutoCombat fit
        super().fit(X, *y)

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Scale image according to combat estimator and save it.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features). Requires the columns needed by the ImageCombat().
            Input data that will be transformed.

        Returns
        -------
            None, save transformed image
        """
        check_is_fitted(self)

        _, list_image_path = self._check_image_path(X)

        logical_mask = self.mask_ == 1  # force the mask to be logical type
        n_voxels_flattened = np.sum(logical_mask)

        # to avoid OOM, apply sequentially the transform for image path
        for i, image_path in enumerate(tqdm(list_image_path, desc="Transform image")):
            to_convert = X.loc[[i]] if isinstance(X, pd.DataFrame) else X[[i]]
            flattened_array = np.zeros((1, n_voxels_flattened)).astype(self.flattened_dtype)
            image_arr, header = load_nifty_volume_as_array(image_path)
            flattened_array[0, :] = image_arr[logical_mask]
            to_convert, features_columns = self._add_voxels_as_features(to_convert.copy(deep=self.copy),
                                                                        flattened_array)
            # run AutoCombat transform
            adjusted_array = super().transform(to_convert)
            adjusted_array = adjusted_array[:, features_columns]
            #
            nifti_out = logical_mask.copy().astype(self.output_dtype)
            nifti_out[logical_mask] = adjusted_array[0, :]
            save_to_nii(im=nifti_out, header=header, output_dir=self.save_path_transform,
                        filename=os.path.basename(image_path))

    def _check_image_path(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[List, List]:

        self.image_path = [self.image_path] if isinstance(self.image_path, (str, int)) else self.image_path

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.copy())

        column_image_path = _check_exist_vars(X, self.image_path).tolist()

        if X[self.image_path[0]].isnull().values.any():
            raise ValueError("An image path is missing in rows")

        # check nii file exit
        X[self.image_path[0]].apply(lambda x: check_is_nii_exist(x))

        list_image_path = X[self.image_path[0]].to_list()

        return column_image_path, list_image_path

    def _add_voxels_as_features(self, X: Union[np.ndarray, pd.DataFrame], flattened_array: np.ndarray):
        """
        Add voxels as features and drop the image_path columns
        :param X: Initial input of ImageCombat
        :param flattened_array: flattened array
        :return: X concatenate with the flattened array
        """

        init_columns = X.shape[1]
        end_columns = init_columns + flattened_array.shape[1]
        features_columns = list(range(init_columns, end_columns))

        if isinstance(X, pd.DataFrame):
            X[self.image_path] = 0.  # create null columns for image path, allows to keep order
            # parameters of columns parameters (because string type)
            X = pd.concat([X.reset_index(drop=True), pd.DataFrame(flattened_array)], axis=1)
            self.features = list(range(flattened_array.shape[1]))
            if (np.unique(X.columns.to_list(), return_counts=True)[1] > 1).any():
                raise ValueError(f"There is a column in the dataframe with the name in the range: "
                                 f"{features_columns[0]} ... {features_columns[-1]} ")
        elif isinstance(X, np.ndarray):
            X[:, 1] = 0.
            X = np.c_[X, flattened_array]
            self.features = features_columns

        return X, features_columns