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
    df = pd.merge(df, df_company[["Symbol", "List"]], how="left", on="Symbol")
    output_cols = ["Date", "Symbol", "List", "stock_price"]
    return df[output_cols]
