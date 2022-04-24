import argparse

import numpy as np
import pandas as pd


class Preprocessor:
    def main(self):
        pass


class CreateHumanPrediction:
    @staticmethod
    def remove_null_lot(input_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        # primary_keys = marifu_lot_no, ymd
        use_cols = list(config.primary_keys.keys()) + list(config.origin_columns)
        df = input_df[].copy()

    @staticmethod
    def get_houkai_rate(input_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = input_df.copy()
        for ryudo_kikaku in config.houkai_rate_settings["ryudo_kikaku_list"]:
            df[f"houkai_rate_{ryudo_kikaku}"] = df[f"marifu_{ryudo_kikaku}"] - df[f"sck_{ryudo_kikaku}"]
        return df

    @staticmethod
    def get_ido_houkai_rate(input_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = input_df.copy()
        window = config.houkai_rate_settings["rolling_window"]
        min_periods = config.houkai_rate_settings["min_periods"]
        for ryudo_kikaku in config.houkai_rate_settings["ryudo_kikaku_list"]:
            df[f"houkai_rate_{ryudo_kikaku}_1s_{window}r"] = df.groupby()[f"houkai_rate_{ryudo_kikaku}"].apply(
                lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean())
        return df


if __name__ == "__main__":
    # 引数受取り
    parser = argparse.ArgumentParser(description="前処理後のテーブルを出力する")
    parser.add_argument("-f", "--flow_id", help="フローID")
    parser.add_argument("-p", "--output_path", help="テーブルをアウトプットするディレクトリのパス")

    # 引数格納
    args = parser.parse_args()
    output_path = args.output_path  # クエリパス
    flow_id = args.flow_id  # フローID
