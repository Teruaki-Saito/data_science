import pandas as pd
import numpy as np


class ConcatTrainTestDf:
    """train & test結合
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, with_log1p_y: bool):
        self.df_train = train.copy()
        self.df_test = test.copy()
        self.with_log1p_y = with_log1p_y

        # 結合前にy列を整える
        self._set_y_column()

    def concat_df(self):
        df_concat = pd.concat([self.df_train, self.df_test], ignore_index=True, sort=False)
        return df_concat

    def _set_y_column(self):
        self.df_test["y"] = np.nan  # 仮の値を入れる

        # log変換
        if self.with_log1p_y:
            self.df_train["log1p_y"] = np.log1p(self.df_train["y"])
            self.df_test["log1p_y"] = np.nan  # 仮の値を入れる

        # data_typeの追加
        self.df_train["data_type"] = "train"
        self.df_test["data_type"] = "test"
