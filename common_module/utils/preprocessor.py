import pandas as pd
import os
import datetime
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple, Dict
from collections.abc import Callable
import category_encoders as ce


class BaseBlock(metaclass=ABCMeta):
    @abstractmethod
    def fit_transform(
        self,
        input: pd.Series
    ):
        raise NotImplementedError()

    @abstractmethod
    def transform(
        self,
        input: pd.Series
    ):
        """サブクラスで実装しないとエラーになる。
        """
        raise NotImplementedError()


class OrdinalEncodingBlock(BaseBlock):
    """カテゴリー変数のエンコーディング
    """
    def __init__(self, target_cols: List[str]) -> None:
        self.target_cols = target_cols
        self.encoder = None

    def fit_transform(
        self,
        input: pd.Series
    ) -> Tuple[Callable, pd.DataFrame]:
        """学習データ用
        ex) "pref": ["Tokyo", None, "Osaka", "Osaka", "Kyoto"] -> "pref": [1, -1, 3, 3, 4]
        ※登場順に番号がつけられるため注意。np.nanとNoneや学習時になかった情報は-1にエンコードされる。

        Args:
            input (pd.Series): _description_

        Returns:
            Tuple[Callable, pd.DataFrame]: _description_
        """
        self.encoder = ce.OrdinalEncoder()
        self.encoder.fit(input[self.target_cols])
        df_output = self.encoder.transform(input[self.target_cols]).add_prefix("OE_")
        return self.encoder, df_output

    def transform(
        self,
        input: pd.Series,
        encoder: Callable
    ) -> pd.DataFrame:
        """テストデータ用
        Nullや学習時になかった情報は-1にエンコードされる。

        Args:
            input (pd.Series): _description_
            encoder (Callable): 学習用で作成したencoderを使用

        Returns:
            pd.DataFrame: _description_
        """
        return encoder.transform(input[self.target_cols]).add_prefix("OE_")


class CountEncodingBlock(BaseBlock):
    """カテゴリのカウント
    """
    def fit_transform(
        self,
        input: pd.Series,
        target_cols: List[str]
    ) -> Tuple[Callable, pd.DataFrame]:
        """学習データ用
        ex) "pref": ["Tokyo", "Osaka", "Osaka", "Osaka", "Kyoto"] -> "pref": [1, 3, 3, 3, 1]
        ※np.nanとNoneは同じものとみなしてカウントする。

        Args:
            input (pd.Series): _description_

        Returns:
            Tuple[Callable, pd.DataFrame]: _description_
        """
        encoder = ce.CountEncoder()
        encoder.fit(input[target_cols])
        df_output = encoder.transform(input[target_cols]).add_prefix("CE_")
        return encoder, df_output

    def transform(
        self,
        input: pd.DataFrame,
        target_cols: List[str],
        encoder: Callable
    ) -> pd.DataFrame:
        """テストデータ用
        学習時になかった情報は0にエンコードされる。

        Args:
            input (pd.Series): _description_
            encoder (Callable): 学習用で作成したencoderを使用

        Returns:
            pd.DataFrame: _description_
        """
        return encoder.transform(input[target_cols]).add_prefix("CE_")


class GroupingBlock(BaseBlock):
    def __init__(
        self,
        group_keys: List[str],
        target_cols: List[str],
        methods: List[str]
    ):
        self.group_keys = group_keys
        self.target_cols = target_cols
        self.methods = methods

    def fit_transform(
        self,
        input: pd.DataFrame
    ) -> Tuple[Callable, pd.DataFrame]:
        """学習データ用
        ※np.nanとNoneは同じものとみなしてカウントする。

        Args:
            input (pd.Series): _description_

        Returns:
            Tuple[Callable, pd.DataFrame]: _description_
        """
        df_agg_set = {}
        df_output = input.copy()
        for group_key in self.group_keys:
            for i, target_col in enumerate(self.target_cols):
                if i == 0:
                    df_agg = self._agg(input, group_key, target_col)
                else:
                    df_agg_part = self._agg(input, group_key, target_col)
                    df_agg = pd.merge(df_agg, df_agg_part, how="left", on=group_key)
            df_output = pd.merge(df_output, df_agg, how="left", on=group_key)
            df_agg_set[group_key] = df_agg
            print(f"{group_key} agg complete")
        return df_agg_set, df_output

    def transform(
        self,
        input: pd.DataFrame,
        df_agg_set: Dict
    ) -> pd.DataFrame:
        """テストデータ用
        学習時になかった情報は0にエンコードされる。

        Args:
            input (pd.Series): _description_
            df_agg (Callable): 学習用で作成したdf_aggを使用

        Returns:
            pd.DataFrame: _description_
        """
        df_output = input.copy()
        for group_key in self.group_keys:
            df_output = pd.merge(df_output, df_agg_set[group_key], how="left", on=group_key)
        return df_output

    def _agg(
        self,
        input: pd.DataFrame,
        group_key: str,
        target_col: str
    ) -> pd.DataFrame:
        input_notnull = input.dropna(how="any")  # 欠損値が一つでも含まれる行を削除
        df_agg_part = input_notnull.groupby(group_key, as_index=False).agg({target_col: self.methods})
        rename_cols = [group_key] + [f"agg_{m}_{group_key}_by_{target_col}" for m in self.methods]
        df_agg_part.columns = rename_cols
        return df_agg_part


class OneHotEncodingBlock(BaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = ce.OneHotEncoder(use_cat_names=True)
        self.encoder.fit(input_df[self.cols])
        return self.transform(input_df[self.cols])

    def transform(self, input_df):
        return self.encoder.transform(input_df[self.cols]).add_prefix("OHE_")