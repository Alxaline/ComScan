# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
Created on: Jan 14, 2021
"""
import os
from typing import List
from typing import Tuple, Union, Sequence

import SimpleITK as sitk
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def check_exist_vars(df: pd.DataFrame, _vars: List) -> np.ndarray:
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


def get_column_index(df: pd.DataFrame, query_cols: List[str]) -> Union[np.ndarray]:
    return df.columns.get_indexer(query_cols)


def column_var_dtype(df: pd.DataFrame, identify_dtypes: Sequence[str] = ("object",)) -> pd.DataFrame:
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


def one_hot_encoder(df: pd.DataFrame, columns: List[str], drop_column: bool = True,
                    inplace: bool = False) -> pd.DataFrame:
    """
    Encoding categorical feature in the dataframe, allow possibility to keep NaN.
    The categorical feature index and name are from cat_var function. These columns need to be "object" dtypes.
    :param df: input dataframe
    :param columns: List of columns to encode
    :param drop_column: Set to True to drop the original column after encoding. Default to True.
    :param inplace: If False, return a copy. Otherwise, do operation inplace and return None
    :return:
        df: new dataframe where columns are one hot encoded
    """

    check_exist_vars(df, columns)

    if not inplace:
        df = df.copy(deep=True)

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        if drop_column:
            df = df.drop(col, axis=1)

    return df


def scaler_encoder(df: pd.DataFrame, columns: List[str], scaler=StandardScaler(),
                   inplace: bool = False) -> pd.DataFrame:
    """
    Apply sklearn scaler to columns.

    :param df: input dataframe
    :param columns: List of columns to encode
    :param scaler: scaler object from sklearn
    :param inplace: If False, return a copy. Otherwise, do operation inplace and return None
    :return df: new dataframe where columns are scaler encoded
    """

    check_exist_vars(df, columns)

    if not inplace:
        df = df.copy(deep=True)

    le = scaler
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df


def tsne(df: pd.DataFrame, columns: List[str], n_components: int = 2, random_state: Union[int, None] = 123,
         n_jobs: Union[int, None] = -1):
    """
    t-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    :param df: input dataframe
    :param columns: List of columns to use
    :param n_components: Dimension of the embedded space. Default 2.
    :param random_state: int, RandomState instance or None, optional, default: 123
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param n_jobs, default=-1
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :return: array-like with projections
    """
    tsne = TSNE(n_components=n_components, random_state=random_state, n_jobs=n_jobs)
    projections = tsne.fit_transform(df[columns])
    return projections


def u_map(df: pd.DataFrame, columns: List[str], n_components: int = 2, random_state: Union[int, None] = 123,
          n_jobs: Union[int, None] = -1):
    """
    Just like t-SNE, UMAP is a dimensionality reduction specifically designed for visualizing complex data in
    low dimensions (2D or 3D). As the number of data points increase, UMAP becomes more time efficient compared to TSNE.

    :param df: input dataframe
    :param columns: List of columns to use
    :param n_components: Dimension of the embedded space. Default 2.
    :param random_state: int, RandomState instance or None, optional, default: 123
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param n_jobs, default=-1
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :return: array-like with projections
    """
    _umap = umap.UMAP(n_components=n_components, init='random', random_state=random_state, n_jobs=n_jobs)
    projections = _umap.fit_transform(df[columns])
    return projections


def split_filename(file_name: str) -> Tuple[str, str, str]:
    """
    Split file_name into folder path name, basename, and extension name.

    :param file_name: full path
    :return: path name, basename, extension name
    """
    pth = os.path.dirname(file_name)
    f_name = os.path.basename(file_name)

    ext = None
    for special_ext in ['.nii.gz']:
        ext_len = len(special_ext)
        if f_name[-ext_len:].lower() == special_ext:
            ext = f_name[-ext_len:]
            f_name = f_name[:-ext_len] if len(f_name) > ext_len else ''
            break
    if not ext:
        f_name, ext = os.path.splitext(f_name)
    return pth, f_name, ext


def check_is_nii_exist(input_file_path: str) -> str:
    """
    Check if a directory exist.

    :param input_file_path: string of the path of the nii or nii.gz.
    :return: string if exist, else raise Error.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"{input_file_path} was not found, check if it's a valid file path")

    pth, fnm, ext = split_filename(input_file_path)
    if ext not in [".nii", ".nii.gz"]:
        raise FileExistsError(f"extension of {input_file_path} need to be '.nii' or '.nii.gz'")
    return input_file_path


def load_nifty_volume_as_array(input_path_file: str) -> Tuple[np.ndarray, Tuple[Tuple, Tuple, Tuple]]:
    """
    Load nifty image into numpy array [z,y,x] axis order.
    The output array shape is like [Depth, Height, Width].

    :param input_path_file: input path file, should be *.nii or *.nii.gz
    :return: a numpy data array, (with header)
    """
    img = sitk.ReadImage(input_path_file)
    data = sitk.GetArrayFromImage(img)

    origin, spacing, direction = img.GetOrigin(), img.GetSpacing(), img.GetDirection()
    return data, (origin, spacing, direction)


def save_to_nii(im: np.ndarray, header: (tuple, tuple, tuple), output_dir: str, filename: str, mode: str = "image",
                gzip: bool = True) -> None:
    """
    Save numpy array to nii.gz format to submit.

    :param im: array numpy
    :param header: header metadata (origin, spacing, direction).
    :param output_dir: Output directory.
    :param filename: Filename of the output file.
    :param mode: save as 'image' or 'label'
    :param gzip: zip nii (ie, nii.gz)
    """
    origin, spacing, direction = header
    if mode == "label":
        img = sitk.GetImageFromArray(im.astype(np.uint8))
    else:
        img = sitk.GetImageFromArray(im.astype(np.float32))
    img.SetOrigin(origin), img.SetSpacing(spacing), img.SetDirection(direction)

    if gzip:
        ext = ".nii.gz"
    else:
        ext = ".nii"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(img, os.path.join(output_dir, filename) + ext)


def mat_to_bytes(nrows: int, ncols: int, dtype: int = 32, out: str = "GB") -> float:
    """
    # https://gist.github.com/dimalik/f4609661fb83e3b5d22e7550c1776b90
    Calculate the size of a numpy array in bytes.

    :param nrows: the number of rows of the matrix.
    :param ncols: the number of columns of the matrix.
    :param dtype: the size of each element in the matrix. Defaults to 32bits.
    :param out: the output unit. Defaults to gigabytes (GB)
    :returns: the size of the matrix in the given unit
    :rtype: a float
    """
    sizes = {v: i for i, v in enumerate("BYTES KB MB GB TB".split())}
    return nrows * ncols * dtype / 8 / 1024. ** sizes[out]
