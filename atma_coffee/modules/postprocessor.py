import pandas as pd
from typing import Dict, List, Union


def ceil_pred_cols(
    input: pd.DataFrame,
    target_cols: List,
    ceil_value: int
) -> pd.DataFrame:
    """特定の目的変数の予測結果について切り上げを行う関数

    Args:
        input (pd.DataFrame): _description_
        target_cols (List): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_output = input.copy()
    for target_col in target_cols:
        df_output[target_col] = df_output[target_col].apply(
            lambda x: 100 if x > ceil_value else x)
    return df_output
