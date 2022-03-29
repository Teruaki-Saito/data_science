from abc import ABCMeta, abstractmethod
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost
from catboost import Pool
from typing import Dict, List, Union

# https://github.com/tubo213/atmacup10_colum2131_tubo/blob/main/src/tubo/module/models.py


class AbsTrainModule(metaclass=ABCMeta):
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        self._feature_columns = config.feature_columns
        self._weight_column = config.weight_column
        self._target = target

    @abstractmethod
    def train_model():
        raise NotImplementedError()


class TrainLGBModule(AbsTrainModule):
    """
    学習に必要なデータを受け取り、モデルを学習するクラス
    """
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        super().__init__(config, target)
        self._hyper_params = config.lgb_hyper_params
        self._train_config = config.lgb_train_config

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        have_weight: bool = False
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
        y_train = df_train[self._target].copy()
        y_val = df_val[self._target].copy()
        X_train = df_train[self._feature_columns.keys()].copy()
        X_val = df_val[self._feature_columns.keys()].copy()

        # LGBM用のデータセット作成
        if have_weight:
            weight_train = calc_weight(input_df=X_train, lgb_weight=self._weight_column)
            weight_val = calc_weight(input_df=X_val, lgb_weight=self._weight_column)
            train_data = lgb.Dataset(X_train, label=y_train, weight=weight_train)
            val_data = lgb.Dataset(X_val, label=y_val, weight=weight_val)
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

        # log
        evals_result = {}
        # 学習
        trained_model = lgb.train(
            params=self._hyper_params,
            train_set=train_data,
            num_boost_round=self._train_config["num_iterations"],
            # valid_sets=val_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            verbose_eval=self._train_config["verbose_eval"],
            early_stopping_rounds=self._train_config["early_stopping"],
            evals_result=evals_result
        )
        # feature importances
        df_imp = pd.DataFrame()
        df_imp["feature"] = X_train.columns
        df_imp["gain"] = trained_model.feature_importance(importance_type="gain")
        return trained_model, df_imp, evals_result


class TrainLGBModuleAdv(AbsTrainModule):
    """
    学習に必要なデータを受け取り、モデルを学習するクラス
    """
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        super().__init__(config, target)
        self._hyper_params = config.AdversarialValidation["lgb_hyper_params"]
        self._train_config = config.lgb_train_config

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        have_weight: bool = False
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
        y_train = df_train[self._target].copy()
        y_val = df_val[self._target].copy()
        X_train = df_train[self._feature_columns.keys()].copy()
        X_val = df_val[self._feature_columns.keys()].copy()

        # LGBM用のデータセット作成
        if have_weight:
            weight_train = calc_weight(input_df=X_train, lgb_weight=self._weight_column)
            weight_val = calc_weight(input_df=X_val, lgb_weight=self._weight_column)
            train_data = lgb.Dataset(X_train, label=y_train, weight=weight_train)
            val_data = lgb.Dataset(X_val, label=y_val, weight=weight_val)
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

        # log
        evals_result = {}
        # 学習
        trained_model = lgb.train(
            params=self._hyper_params,
            train_set=train_data,
            num_boost_round=self._train_config["num_iterations"],
            valid_sets=val_data,
            # valid_sets=[train_data, val_data],
            verbose_eval=self._train_config["verbose_eval"],
            early_stopping_rounds=self._train_config["early_stopping"],
            evals_result=evals_result
        )
        # feature importances
        df_imp = pd.DataFrame()
        df_imp["feature"] = X_train.columns
        df_imp["gain"] = trained_model.feature_importance(importance_type="gain")
        return trained_model, df_imp, evals_result


class TrainCABModule(AbsTrainModule):
    """
    学習に必要なデータを受け取り、モデルを学習するクラス
    """
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        super().__init__(config, target)
        self._hyper_params = config.cat_hyper_params
        self._train_config = config.cat_train_config

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        have_weight: bool = False
    ):
        """
        学習を行う

        Args:
            df_train ([type]): 学習用データフレーム
            df_val ([type]): Validation用データフレーム

        Returns:
            trained_model(model): 学習済みモデル
        """
        # yを取り出す
        y_train = df_train[self._target].copy()
        y_val = df_val[self._target].copy()
        X_train = df_train[self._feature_columns.keys()].copy()
        X_val = df_val[self._feature_columns.keys()].copy()

        # CatB用のデータセット作成
        if have_weight:
            weight_train = calc_weight(input_df=X_train, lgb_weight=self._weight_column)
            weight_val = calc_weight(input_df=X_val, lgb_weight=self._weight_column)
            train_data = Pool(X_train, label=y_train, cat_features=self._train_config["cat_features"], weight=weight_train)
            val_data = Pool(X_val, label=y_val, cat_features=self._train_config["cat_features"], weight=weight_val)
        else:
            train_data = Pool(X_train, label=y_train, cat_features=self._train_config["cat_features"])
            val_data = Pool(X_val, label=y_val, cat_features=self._train_config["cat_features"])

        # 学習
        model = CatBoost(self._hyper_params)
        model.fit(
            train_data,
            plot=self._train_config["plot"],
            eval_set=[val_data],
            verbose=self._train_config["verbose"]
        )
        # feature importances
        df_imp = pd.DataFrame()
        df_imp["feature"] = X_train.columns
        df_imp["importance"] = model.get_feature_importance()
        return model, df_imp


class TrainXGBModule(AbsTrainModule):
    """
    学習に必要なデータを受け取り、モデルを学習するクラス
    """
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        super().__init__(config, target)
        self._hyper_params = config.xgb_hyper_params
        self._train_config = config.xgb_train_config

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        have_weight: bool = False
    ):
        """
        学習を行う

        Args:
            df_train ([type]): 学習用データフレーム
            df_val ([type]): Validation用データフレーム

        Returns:
            trained_model(model): 学習済みモデル
        """
        # yを取り出す
        y_train = df_train[self._target].copy()
        y_val = df_val[self._target].copy()
        X_train = df_train[self._feature_columns.keys()].copy()
        X_val = df_val[self._feature_columns.keys()].copy()

        # CatB用のデータセット作成
        if have_weight:
            weight_train = calc_weight(input_df=X_train, lgb_weight=self._weight_column)
            weight_val = calc_weight(input_df=X_val, lgb_weight=self._weight_column)
            train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=self._train_config["enable_categorical"], weight=weight_train)
            val_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=self._train_config["enable_categorical"], weight=weight_val)
        else:
            train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=self._train_config["enable_categorical"])
            val_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=self._train_config["enable_categorical"])

        # 学習の記録
        evals_result = {}

        # 学習
        trained_model = xgb.train(
            params=self._hyper_params,
            dtrain=train_data,
            num_boost_round=self._train_config["num_boost_round"],
            early_stopping_rounds=self._train_config["early_stopping_rounds"],
            evals=[(train_data, "train"), (val_data, "valid")],
            evals_result=evals_result,
            verbose_eval=self._train_config["verbose_eval"]
        )

        # feature importances
        dict_imp = trained_model.get_score(importance_type='gain')
        df_imp = pd.DataFrame(list(dict_imp.items()), columns=['feature', 'importance'])
        return trained_model, df_imp, evals_result


def calc_weight(input_df: pd.DataFrame, lgb_weight: str) -> pd.DataFrame:
    """学習時のweightを計算する関数

    Args:
        input_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    weight = 1 / input_df[lgb_weight].replace(0, 1)
    return weight
