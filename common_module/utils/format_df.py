import pandas as pd


def format_df(
    df_: pd.DataFrame,
    dtype_definition: dict
) -> pd.DataFrame:
    """dfの指定列を抽出し、カラムを一括で変換する

    Args:
        df(pd.DataFrame): 元となるデータフレーム
        dtype_definition (dict): カラム、型定義が書かれた辞書
            key：カラム名
            value：データ型

    Returns:
        pd.DataFrame: 整形後のデータフレーム
    """
    df = df_.copy()
    for column, column_type in dtype_definition.items():
        df[f"{column}"] = convert_column_type(df[f"{column}"], column_type)
    return df


def convert_column_type(
    column: pd.Series,
    column_type: str
) -> pd.Series:
    """各列の型を指定された型に変換する

    Args:
        column (pd.Series): 変換対象となるSeries
        column_type (str): 変換後の型

    Returns:
        pd.Series: 変換後のSeries
    """
    if column_type == "date":
        return pd.to_datetime(column).dt.date
    elif column_type == "datetime":
        return pd.to_datetime(column)
    else:
        return column.astype(column_type)