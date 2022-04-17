import pandas as pd
import numpy as np


def create_minor_categorical_features(
    dataframe: pd.DataFrame,
    target: str,
    threshold: int = 500
) -> pd.DataFrame:
    """マイナーなカテゴリを含む列の特徴量を作成する関数
    ACTIVITY_NAME, COUNTRY_NAME, CURRENCY, TOWN_NAME

    Args:
        dataframe (pd.DataFrame): _description_
        target (str): _description_
        threshold (int, optional): _description_. Defaults to 500.

    Returns:
        pd.DataFrame: _description_
    """
    df = dataframe[["LOAN_ID", target]].copy()
    df[f"create_{target}_count"] = df.groupby(target)["LOAN_ID"].transform("count")
    df[f"fixed_{target}"] = df.apply(
        lambda x: x[target] if x[f"create_{target}_count"] > threshold else "Others", axis=1)
    df = df.drop(target, axis=1)
    return df


def create_comb_categorical_features(
    dataframe: pd.DataFrame,
    target_col1: str,
    target_col2: str
) -> pd.DataFrame:
    df = dataframe[["LOAN_ID", target_col1, target_col2]].copy()
    df[f"create_{target_col1}_{target_col2}"] = df.apply(lambda x: f"{x[target_col1]}_{x[target_col2]}")
    return df


