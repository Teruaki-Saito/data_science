import os
import gc
import datetime
import pickle

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from catboost import Pool


def make_pred_df(model, target, df_test, config, is_test, cv_num=None):
    """
    モデルの予測値を格納したdfを作成する

    Args：
        model (model)：学習済みモデル
        df_test (pd.DataFrame)：予測対象期間のdf
        df_rday (pd.DataFrame)：rdayに紐づいた特徴量のdf（移動平均とscaleがある）
    Returns：
        df_pred (pd.DataFrame)：予測値を格納したdf
    """
    # test時とvalid時の違いを定義
    if is_test:
        drop_cols = config.primary_key
    else:
        drop_cols = config.primary_key + config.pred_cols

    # 予測
    print(f"pred {target} ...")
    y_pred = model.predict(df_test.drop(drop_cols, axis=1))
    # 予測値をdfに書き込む
    pred_col = f"pred_{target}"
    df_pred = df_test[config.primary_key].copy()
    df_pred["cv_num"] = cv_num
    df_pred[pred_col] = y_pred
    # df_pred[pred_col] = df_pred[pred_col].clip(lower=0)  # 予測値の最小値は0にする
    return df_pred


class PredictModule:
    def __init__(
        self,
        config: dict,
        target: str,
        model_type: str,
        is_test: bool,
        clip: bool = False
    ) -> None:
        """コンストラクタ

        Args:
            config (dict): parameter等を記載したconfigファイル
            model_type (str): lightgbm, xgboost, catboostから選択
            is_test (bool): _description_
            clip (bool, optional): _description_. Defaults to False.
        """
        self._primary_key = list(config.primary_key.keys())  # TypeError: unsupported operand type(s) for +: 'dict_keys' and 'list'
        self._feature_columns = list(config.feature_columns.keys())
        self._model_type = model_type
        self._is_test = is_test
        self._clip = clip
        self._target = target

        # モデル別にtrain_configを読みこむ
        if self._model_type == "lightgbm":
            self._train_config = config.lgb_train_config
        elif self._model_type == "xgboost":
            self._train_config = config.xgb_train_config
        elif self._model_type == "catboost":
            self._train_config = config.cat_train_config

    def predict(self, model, test_df: pd.DataFrame, cv_num: int = 1) -> pd.DataFrame:
        # baseとなるdfの作成
        df_output = test_df[self._primary_key].copy()
        X_test = test_df[self._feature_columns].copy()

        # log
        print(f"pred {self._target} ...")
        # model_type別に処理
        if self._model_type == "lightgbm":
            test_data = X_test.copy()  # 変換不要（Cannot use Dataset instance for prediction, please use raw data instead)
        elif self._model_type == "xgboost":
            test_data = xgb.DMatrix(X_test, enable_categorical=self._train_config["enable_categorical"])
        elif self._model_type == "catboost":
            test_data = Pool(X_test, cat_features=self._train_config["cat_features"])
        # 予測
        y_pred = model.predict(test_data)
        # 予測値をdfに書き込む
        pred_col = f"pred_{self._target}"
        df_output["cv_num"] = cv_num
        df_output[pred_col] = y_pred
        # clipするかどうか
        if self._clip:
            df_output[pred_col] = df_output[pred_col].clip(lower=0)  # 予測値の最小値は0にする
        return df_output
