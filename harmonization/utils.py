# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
Created on: Jan 14, 2021
"""
import os
from typing import Tuple, List, Union, Sequence

import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from harmonization.neurocombat import _check_exist_vars


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


def one_hot_encoder(df: pd.DataFrame, columns: List[str], drop_column: bool = True) -> pd.DataFrame:
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


def scaler_encoder(df: pd.DataFrame, columns: List[str], scaler=StandardScaler()) -> pd.DataFrame:
    """
    Apply sklearn scaler to columns.
    :param df: input dataframe
    :param columns: List of columns to encode
    :param scaler: scaler object from sklearn
    :return df: new dataframe where columns are scaler encoded
    """

    _check_exist_vars(df, columns)

    le = scaler
    df[columns] = le.fit_transform(df[columns])

    return df


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
