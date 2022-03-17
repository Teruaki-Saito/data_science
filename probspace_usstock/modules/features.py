import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def create_company_features(
    input: pd.DataFrame,
    duplicate_companies: List[str]
) -> pd.DataFrame:
    """companyデータを使った特徴量を作成する関数

    Args:
        input (pd.DataFrame): _description_
        duplicate_companies (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # baseのdfを作成
    df_base = input[["Symbol", "IPOyear", "Sector"]].drop_duplicates()
    # listで重複上場している会社があるためリストを修正
    df_list = modify_list_column(input, duplicate_companies)
    df_output = pd.merge(df_base, df_list, how="left", on="Symbol")
    return df_output


def modify_list_column(
    input: pd.DataFrame,
    duplicate_companies: List[str]
) -> pd.DataFrame:
    """Listで重複上場している会社があるため修正する関数

    Args:
        input (pd.DataFrame): _description_
        duplicate_companies (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_list = input[["Symbol", "List"]].copy()
    for k in duplicate_companies.keys():
        df_list.loc[df_list["Symbol"].isin(duplicate_companies[k]), "List"] = k
    df_output = df_list[["Symbol", "List"]].drop_duplicates()
    return df_output


def calc_Symbol_lag_features(input: pd.DataFrame) -> pd.DataFrame:
    """Symbol粒度のlag特徴量を計算する関数

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_lag = input[["Date", "Symbol", "stock_price"]].copy()

    group_key = ["Symbol"]

    # 1~4週間前
    shift_range = [1, 2, 3, 4]
    for n in shift_range:
        df_lag[f"stock_price_{n}s"] = df_lag.groupby(group_key)["stock_price"].apply(
            lambda x: x.shift(n))

    # 差分
    def calc_shift_diff(x: pd.Series) -> pd.Series:
        diff = x[f"stock_price_{d-1}s"] - x[f"stock_price_{d}s"]
        if diff > 0:
            return 1
        elif diff <= 0:
            return 0
        else:
            return np.nan

    for d in shift_range[1:]:
        df_lag[f"increase_stock_price_{d-1}s{d}s"] = df_lag.apply(lambda x: calc_shift_diff(x), axis=1)

    # 移動平均系の特徴量
    window_list = [4, 8, 12]
    methods = ["mean", "median", "max", "min"]
    for window in window_list:
        _feature_columns = [f"stock_price_1s_{window}r_{method}" for method in methods]
        df_lag[_feature_columns] = df_lag.groupby(group_key)["stock_price"].apply(
            lambda x: x.shift(1).rolling(window=window).agg(methods))

    # pred_colsを削除
    df_lag = df_lag.drop("stock_price", axis=1)
    return df_lag


# def calc_Symbol_lag_features(input: pd.DataFrame) -> pd.DataFrame:
#     """Symbol粒度のlag特徴量を計算する関数

#     Args:
#         input (pd.DataFrame): _description_

#     Returns:
#         pd.DataFrame: _description_
#     """
#     df_lag = input[["Date", "Symbol", "List", "stock_price"]].copy()

#     # stock_price_log1p_diffが目的変数
#     df_lag["stock_price_log1p"] = np.log1p(df_lag["stock_price"])
#     df_lag["stock_price_log1p_1s"] = df_lag.groupby("Symbol")["stock_price_log1p"].apply(lambda x: x.shift(1))
#     df_lag["stock_price_log1p_diff"] = df_lag["stock_price_log1p"] - df_lag["stock_price_log1p_1s"]

#     # 1週間前, 2週間前
#     df_lag["stock_price_log1p_diff_1s"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(lambda x: x.shift(1))
#     df_lag["stock_price_log1p_diff_2s"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(lambda x: x.shift(2))

#     # mean
#     df_lag["stock_price_log1p_diff_1s_4r"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(
#         lambda x: x.shift(1).rolling(window=4).mean())
#     df_lag["stock_price_log1p_diff_1s_8r"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(
#         lambda x: x.shift(1).rolling(window=8).mean())
#     df_lag["stock_price_log1p_diff_1s_12r"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(
#         lambda x: x.shift(1).rolling(window=12).mean())

#     # median
#     df_lag["stock_price_log1p_diff_1s_4r_med"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(
#         lambda x: x.shift(1).rolling(window=4).median())
#     df_lag["stock_price_log1p_diff_1s_8r_med"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(
#         lambda x: x.shift(1).rolling(window=8).median())
#     df_lag["stock_price_log1p_diff_1s_12r_med"] = df_lag.groupby("Symbol")["stock_price_log1p_diff"].apply(
#         lambda x: x.shift(1).rolling(window=12).median())
#     return df_lag


# def calc_List_lag_features(input: pd.DataFrame) -> pd.DataFrame:
#     """Date&List粒度のlag特徴量を作成する関数

#     Args:
#         df_Symbol_lag (pd.DataFrame): df_List_lag

#     Returns:
#         pd.DataFrame: _description_
#     """
#     df = input[["Date", "Symbol", "List", "stock_price_log1p_diff_1s"]].copy()
#     df_list_lag = df.groupby(["Date", "List"])["stock_price_log1p_diff_1s"].agg(["mean", "median"]).reset_index()
#     df_list_lag.columns = ["Date", "List", "stock_price_log1p_diff_1s_List_mean", "stock_price_log1p_diff_1s_List_med"]
#     df_list_lag[["Date", "List"]] = df_list_lag[["Date", "List"]].astype(object)  # reduce_mem_usageでcategoryだとエラーになるため
#     return df_list_lag


def create_ymd_features(input: pd.DataFrame) -> pd.DataFrame:
    """ymd関連の特徴量を作成する関数

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_ymd = pd.DataFrame()
    df_ymd["Date"] = input["Date"].unique()
    df_ymd["Date_datetime"] = pd.to_datetime(df_ymd["Date"])
    df_ymd["year"] = df_ymd["Date_datetime"].dt.year
    df_ymd["month"] = df_ymd["Date_datetime"].dt.month
    df_ymd["day"] = df_ymd["Date_datetime"].dt.day
    df_ymd["week_of_month"] = df_ymd["Date_datetime"].dt.day.map(calc_week_no)  # 週番号（月）
    df_ymd["week_of_year"] = df_ymd["Date_datetime"].dt.isocalendar().week.astype(int)  # 週番号(年)
    df_ymd = df_ymd.drop("Date_datetime", axis=1)
    return df_ymd


def calc_week_no(day):
    """
    週番号を返す関数

    Args:
        day (date):日付（何日目か）
    Returns:
        week_no (int)：週番号
    """
    week_no = (day - 1) // 7 + 1
    return week_no
