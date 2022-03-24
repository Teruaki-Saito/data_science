import lightgbm as lgb
import optuna.integration.lightgbm as LGB_optuna

import pandas as pd
import numpy as np

from probspace_usstock.modules.train_module import calc_weight


class OptunaParameterTuning:
    """学習に必要なデータを受け取り、モデル学習用のハイパーパラメータをチューニングするクラス
    以下7つのパラメータをチューニングできる
        - lambda_l1
        - lambda_l2
        - num_leaves
        - feature_fraction
        - bagging_fraction
        - bagging_freq
        - min_child_samples
    """
    def __init__(
        self, 
        config: dict,
        target: str
    ) -> None:
        """_summary_

        Args:
            config (dict): _description_
            target (str): _description_
        """
        self.__primary_key = config.primary_key
        self.__feature_columns = config.feature_columns
        self.__lgb_optuna_config = config.lgb_optuna_config
        self.__lgb_weight_column = config.lgb_weight_column
        self.__target = target

    def tune_model_parameter(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        have_weight: bool = False
    ) -> dict:
        """
        学習を行う

        Args:
            df_train ([type]): 学習用データフレーム
            df_val ([type]): Validation用データフレーム
            weight_train ([type]): 学習データの各行の重み
            weight_val ([type]): 検証データの各行の重み

        Returns:
            trained_model(model): 学習済みモデル
        """
        # yを取り出す
        y_train = df_train[self.__target].copy()
        y_val = df_val[self.__target].copy()
        X_train = df_train[self.__feature_columns.keys()].drop(self.__primary_key, axis=1).copy()
        X_val = df_val[self.__feature_columns.keys()].drop(self.__primary_key, axis=1).copy()

        # LGBM用のデータセット作成
        if have_weight:
            weight_train = calc_weight(input_df=X_train, lgb_weight=self.__lgb_weight_column)
            weight_val = calc_weight(input_df=X_val, lgb_weight=self.__lgb_weight_column)
            train_data = lgb.Dataset(X_train, label=y_train, weight=weight_train)
            val_data = lgb.Dataset(X_val, label=y_val, weight=weight_val)
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

        # チューニング
        tuned_model = LGB_optuna.train(
            params=self.__lgb_optuna_config["fixed_params"],
            train_set=train_data,
            valid_sets=val_data,
            verbose_eval=self.__lgb_optuna_config["verbose_eval"],
            early_stopping_rounds=self.__lgb_optuna_config["early_stopping_rounds"],
        )
        # 最適なパラメータの表示
        best_params = tuned_model.params
        print("Best params:", best_params)

        return best_params
