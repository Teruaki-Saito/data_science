import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Union


class TrainLGBModule:
    """
    学習に必要なデータを受け取り、モデルを学習するクラス

        Args:
            training_data(dict): 学習に必要なデータがn_shift分格納された辞書
            target (str): 目的変数
            hyper_params (Dict[str, Union[str, int]]): モデルのパラメータ
            train_config (Dict[str, Union[int, List[str]]]): 訓練パラメータ

        Attributes:
            __training_data_list (list): 学習に必要なデータがn_shift分格納されたリスト
            trained_model_list(list): 学習済みモデルを格納する
            __target(str): 目的変数
            __params(dict): モデルのパラメータ
            __params_train(dict): 訓練パラメータ
    """
    def __init__(
        self, 
        config: dict,
        target: str
    ) -> None:
        self.__primary_key = config.primary_key
        self.__feature_columns = config.feature_columns
        self.__hyper_params = config.lgb_hyper_params
        self.__train_config = config.lgb_train_config
        self.__target = target

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        have_weight: bool = False,
        weight_train: pd.Series = None,
        weight_val: pd.Series = None
    ):
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
            train_data = lgb.Dataset(X_train, label=y_train, weight=weight_train)
            val_data = lgb.Dataset(X_val, label=y_val, weight=weight_val)
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

        # log
        evals_result = {}
        # 学習
        trained_model = lgb.train(
            params=self.__hyper_params,
            train_set=train_data,
            num_boost_round=self.__train_config["num_iterations"],
            valid_sets=[train_data, val_data],
            verbose_eval=self.__train_config["verbose_eval"],
            early_stopping_rounds=self.__train_config["early_stopping"],
            evals_result=evals_result
        )
        # feature importances
        df_imp = pd.DataFrame()
        df_imp["feature"] = X_train.columns
        df_imp["gain"] = trained_model.feature_importance(importance_type="gain")
        return trained_model, df_imp, evals_result
