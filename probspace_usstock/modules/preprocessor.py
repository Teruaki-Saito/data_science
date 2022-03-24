import pandas as pd
import numpy as np


def melt_dataframe(input: pd.DataFrame) -> pd.DataFrame:
    """横持ちを縦持ちに変換する関数

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    symbol_list = input.columns[1:]  # 0はDate
    df = pd.melt(
        input,
        id_vars="Date",
        value_vars=symbol_list,
        value_name="stock_price"
    )
    df = df.rename(columns={"variable": "Symbol"})  # defaultでvariableになるため
    df = df[df["Date"].notnull()]  # Dateがnullで他も全てnullの行があるため
    return df


def create_base_dataframe(
    input: pd.DataFrame,
    df_company: pd.DataFrame
) -> pd.DataFrame:
    """ラグ特徴量を作成するために必要な情報を追加する関数

    Args:
        input (pd.DataFrame): _description_
        df_company (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_to20191117 = input[["Date", "Symbol", "stock_price"]].copy()

    # 予測日用のdataframe
    df_20191124 = pd.DataFrame()
    df_20191124["Symbol"] = input["Symbol"].unique()
    df_20191124["Date"] = "2019/11/24"
    df_20191124["stock_price"] = np.nan

    df = pd.concat([df_to20191117, df_20191124], ignore_index=True, sort=False)
    df = pd.merge(df, df_company[["Symbol", "List", "Sector", "Industry"]], how="left", on="Symbol")
    return df


def calc_seasonal_decompose(input_df: pd.DataFrame, ymd_df: pd.DataFrame, period: int = 53):
    """トレンド、季節成分、残差を求める関数
    ※残差は目的変数とする

    Args:
        input_df (pd.DataFrame): _description_
        period (int, optional): _description_. Defaults to 53.

    Returns:
        _type_: _description_
    """
    df = input_df[["Symbol", "Date", "stock_price"]].copy()
    df_woy = ymd_df[["Date", "week_of_year"]].copy()
    df = pd.merge(df, df_woy, how="left", on="Date")

    # トレンドを求める
    df["trend"] = df.groupby(["Symbol"])["stock_price"].apply(
        lambda x: x.shift(1).rolling(window=period).mean())
    df["diff"] = df["stock_price"] - df["trend"]

    # 季節成分を求める
    df_seasonal = df.groupby(["Symbol", "week_of_year"], as_index=False)["diff"].mean()
    df_seasonal.columns = ["Symbol", "week_of_year", "seasonality"]
    # merge
    df_output = pd.merge(df, df_seasonal, how="left", on=["Symbol", "week_of_year"])

    # 残差を求める
    df_output["residual"] = df_output["diff"] - df_output["seasonality"]
    output_cols = ["Symbol", "Date", "week_of_year", "trend", "seasonality", "residual"]
    return df_output[output_cols]
