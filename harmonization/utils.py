from typing import List, Union, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from harmonization.neurocombat import _check_exist_vars


def get_column_index(df: pd.DataFrame, query_cols: List[str]) -> Union[np.ndarray]:
    return df.columns.get_indexer(query_cols)


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