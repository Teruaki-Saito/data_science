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
    # listで重複上場している会社があるためリストを修正
    df_list = modify_list_column(input, duplicate_companies)
    # sector列の修正
    df_sector = modify_sector_column(input)
    # industry列の修正
    df_industry = modify_industry_column(input)
    # baseのdfを作成し、マージ
    df_base = input[["Symbol", "IPOyear"]].drop_duplicates()
    df_output = pd.merge(df_base, df_list, how="left", on="Symbol")
    df_output = pd.merge(df_output, df_sector, how="left", on="Symbol")
    df_output = pd.merge(df_output, df_industry, how="left", on="Symbol")
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


def modify_sector_column(
    input: pd.DataFrame
) -> pd.DataFrame:
    """sector列の前処理を行う
    null -> unknownとする

    Args:
        input (pd.DataFrame): _description_
        duplicate_companies (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = input[["Symbol", "Sector"]].drop_duplicates()
    # nullはunknownとしておく
    df["Sector"] = df["Sector"].fillna("unknown")
    return df


def modify_industry_column(
    input: pd.DataFrame
) -> pd.DataFrame:
    """industry列の前処理を行う
    ・最低でも1分岐で5000行ほど確保したいので、2000 // 417 = 5よりも該当会社が少ないものはまとめる
    ・null -> unknownとする

    Args:
        input (pd.DataFrame): _description_
        duplicate_companies (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = input[["Symbol", "Industry"]].drop_duplicates()
    # マイナーなindustryはminor_industryにまとめる
    df_ind = df["Industry"].value_counts().reset_index()  # index列: Industry名, Industry列: 該当数
    minor_industry_list = df_ind[df_ind["Industry"] < 5]["index"].unique().tolist()
    df.loc[df["Industry"].isin(minor_industry_list), "Industry"] = "minor_industry"
    # nullはunknownとしておく
    df["Industry"] = df["Industry"].fillna("unknown")
    return df


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
    window_list = [4, 8, 12, 24, 36]
    methods = ["mean", "median", "max", "min"]
    for window in window_list:
        _feature_columns = [f"stock_price_1s_{window}r_{method}" for method in methods]
        df_lag[_feature_columns] = df_lag.groupby(group_key)["stock_price"].apply(
            lambda x: x.shift(1).rolling(window=window).agg(methods))

    # pred_colsを削除
    df_lag = df_lag.drop("stock_price", axis=1)
    return df_lag


def calc_Symbol_lag_log_features(input: pd.DataFrame) -> pd.DataFrame:
    """Symbol粒度のlag特徴量を計算する関数(log変換ver)

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_lag = input[["Date", "Symbol", "stock_price"]].copy()

    group_key = ["Symbol"]

    # logに変換
    df_lag["stock_price_log1p"] = np.log1p(df_lag["stock_price"])

    # 1~4週間前
    shift_range = [1, 2, 3, 4]
    for n in shift_range:
        df_lag[f"stock_price_log1p_{n}s"] = df_lag.groupby(group_key)["stock_price_log1p"].apply(
            lambda x: x.shift(n))

    # 差分
    def calc_shift_diff(x: pd.Series) -> pd.Series:
        diff = x[f"stock_price_log1p_{d-1}s"] - x[f"stock_price_log1p_{d}s"]
        if diff > 0:
            return 1
        elif diff <= 0:
            return 0
        else:
            return np.nan

    for d in shift_range[1:]:
        df_lag[f"increase_stock_price_log1p_{d-1}s{d}s"] = df_lag.apply(lambda x: calc_shift_diff(x), axis=1)

    # 移動平均系の特徴量
    window_list = [4, 8, 12]
    methods = ["mean", "median", "max", "min"]
    for window in window_list:
        _feature_columns = [f"stock_price_log1p_1s_{window}r_{method}" for method in methods]
        df_lag[_feature_columns] = df_lag.groupby(group_key)["stock_price_log1p"].apply(
            lambda x: x.shift(1).rolling(window=window).agg(methods))

    # pred_colsを削除
    df_lag = df_lag.drop(["stock_price"], axis=1)
    return df_lag


def calc_Symbol_lag_log1pRate_features(input: pd.DataFrame) -> pd.DataFrame:
    """Symbol粒度のlag特徴量を計算する関数(log変換 + diff)

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_lag = input[["Date", "Symbol", "List", "Sector", "Industry", "stock_price"]].copy()

    group_key = ["Symbol"]

    # lag rate(対数収益率)
    df_lag["stock_price_1s"] = df_lag.groupby(group_key)["stock_price"].apply(lambda x: x.shift(1))
    df_lag["stock_price_log1pRate"] = np.log1p(df_lag["stock_price"] / df_lag["stock_price_1s"])

    # 1~4週間前
    shift_range = [1, 2, 3, 4]
    for n in shift_range:
        df_lag[f"stock_price_log1pRate_{n}s"] = df_lag.groupby(group_key)["stock_price_log1pRate"].apply(
            lambda x: x.shift(n))

    # # 差分
    # def calc_shift_diff(x: pd.Series) -> pd.Series:
    #     diff = x[f"stock_price_log1pRate_{d-1}s"] - x[f"stock_price_log1pRate_{d}s"]
    #     if diff > 0:
    #         return 1
    #     elif diff <= 0:
    #         return 0
    #     else:
    #         return np.nan

    # for d in shift_range[1:]:
    #     df_lag[f"increase_stock_price_log1pRate_{d-1}s{d}s"] = df_lag.apply(lambda x: calc_shift_diff(x), axis=1)

    # 移動平均系の特徴量
    window_list = [4, 8, 12, 24, 53]
    methods = ["mean", "median", "max", "min"]
    for window in window_list:
        _feature_columns = [f"stock_price_log1pRate_1s_{window}r_{method}" for method in methods]
        df_lag[_feature_columns] = df_lag.groupby(group_key)["stock_price_log1pRate"].apply(
            lambda x: x.shift(1).rolling(window=window).agg(methods))

    # pred_colsを削除
    del_cols = ["stock_price"]
    df_lag = df_lag.drop(del_cols, axis=1)

    print(f"end making Symbol lag ...")
    return df_lag


def calc_Symbol_lag_residual_features(input: pd.DataFrame) -> pd.DataFrame:
    """Symbol粒度のlag特徴量を計算する関数(log変換 + diff)

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_lag = input[["Date", "Symbol", "List", "Sector", "Industry", "residual"]].copy()

    group_key = ["Symbol"]

    # 1~4週間前
    shift_range = [1, 2, 3, 4]
    for n in shift_range:
        df_lag[f"residual_{n}s"] = df_lag.groupby(group_key)["residual"].apply(
            lambda x: x.shift(n))

    # 移動平均系の特徴量
    window_list = [4, 8, 12, 24, 53]
    methods = ["mean", "median", "max", "min"]
    for window in window_list:
        _feature_columns = [f"residual_1s_{window}r_{method}" for method in methods]
        df_lag[_feature_columns] = df_lag.groupby(group_key)["residual"].apply(
            lambda x: x.shift(1).rolling(window=window).agg(methods))

    # pred_colsを削除
    del_cols = ["residual"]
    df_lag = df_lag.drop(del_cols, axis=1)

    print(f"end making Symbol lag ...")
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


def calc_category_lag_log1pRate_features(
    input: pd.DataFrame,
    category: str,
    target_cols: List[str],
    methods: List[str],
) -> pd.DataFrame:
    """Date&List粒度のlag特徴量を作成する関数

    Args:
        df_Symbol_lag (pd.DataFrame): df_Symbol_lag

    Returns:
        pd.DataFrame: _description_
    """
    group_key = ["Date", category]
    df = input[group_key + target_cols].copy()

    # grouping
    df_cat_lag = df.groupby(group_key)[target_cols].agg(methods).reset_index()
    df_cat_lag.columns = group_key + [f"{target}_{category}_{method}" for target in target_cols for method in methods]

    print(f"end making {category} lag ...")
    return df_cat_lag


def calc_category_lag_residual_features(
    input: pd.DataFrame,
    category: str,
    target_cols: List[str],
    methods: List[str],
) -> pd.DataFrame:
    """Date&List粒度のlag特徴量を作成する関数

    Args:
        df_Symbol_lag (pd.DataFrame): df_Symbol_lag

    Returns:
        pd.DataFrame: _description_
    """
    group_key = ["Date", category]
    df = input[group_key + target_cols].copy()

    # grouping
    df_cat_lag = df.groupby(group_key)[target_cols].agg(methods).reset_index()
    df_cat_lag.columns = group_key + [f"{target}_{category}_{method}" for target in target_cols for method in methods]

    print(f"end making {category} lag ...")
    return df_cat_lag


def calc_std_features(input: pd.DataFrame) -> pd.DataFrame:
    # 初期設定
    group_key = ["Date", "Symbol"]
    target_cols = ["trend", "seasonality"]
    methods = ["mean", "std"]
    df = input[group_key + target_cols].copy()

    # 標準偏差と平均を求める
    df_stats = df.groupby("Symbol")[target_cols].agg(methods).reset_index()
    df_stats.columns = ["Symbol"] + [f"{target}_{method}" for target in target_cols for method in methods]
    df = pd.merge(df, df_stats, how="left", on="Symbol")

    # 標準化する
    df["trend_standard"] = (df["trend"] - df["trend_mean"]) / df["trend_std"]
    df["seasonality_standard"] = (df["seasonality"] - df["seasonality_mean"]) / df["seasonality_std"]
    output_cols = group_key + ["trend_standard", "seasonality_standard"]
    return df[output_cols]


def create_percentile_flg_columns(input: pd.DataFrame) -> pd.DataFrame:
    """stock_priceの値が極端に小さいまたは大きいものがうまく予測できていないので、フラグを作る関数

    Args:
        input (pd.DataFrame): df_Symbol_lag

    Returns:
        pd.DataFrame: _description_
    """
    stock_price_columns = ["stock_price_log1p_1s_4r_min", "stock_price_log1p_1s_4r_max"]
    df_flg = input[["Date", "Symbol"] + stock_price_columns].copy()
    df_flg["is_05_percentile_stock_price"] = (df_flg["stock_price_log1p_1s_4r_min"] < 1.2).astype(int)
    df_flg["is_95_percentile_stock_price"] = (df_flg["stock_price_log1p_1s_4r_max"] > 4.7).astype(int)
    return df_flg.drop(stock_price_columns, axis=1)


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
