from abc import ABCMeta, abstractmethod
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge
)


class AbsTrainModule(metaclass=ABCMeta):
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        self._feature_columns = list(config.feature_columns.keys())
        self._target = target

    @abstractmethod
    def train_model():
        raise NotImplementedError()


class TrainLRModule(AbsTrainModule):
    """
    学習に必要なデータを受け取り、モデルを学習するクラス
    """
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        super().__init__(config, target)

    def train_model(
        self,
        train_data: pd.DataFrame
    ):
        """学習を行う関数

        Args:
            train_data (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        # yを取り出す
        y_train = train_data[self._target].copy()
        X_train = train_data[self._feature_columns].copy()

        # 重回帰モデルを作成
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 切片と係数を取得
        coef = model.coef_
        intercept = model.intercept_
        return model, coef, intercept


class TrainLassoModule(AbsTrainModule):
    """
    学習に必要なデータを受け取り、モデルを学習するクラス
    """
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        super().__init__(config, target)
        self._lasso_params = config.lasso_params

    def train_model(
        self,
        train_data: pd.DataFrame
    ):
        """学習を行う関数

        Args:
            train (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        # yを取り出す
        y_train = train_data[self._target].copy()
        X_train = train_data[self._feature_columns.keys()].copy()

        # 重回帰モデルを作成
        model = Lasso(alpha=self._lasso_params["alpha"])
        model.fit(X_train, y_train)

        # 切片と係数を取得
        coef = model.coef_
        intercept = model.intercept_
        return model, coef, intercept


class TrainRidgeModule(AbsTrainModule):
    """
    学習に必要なデータを受け取り、モデルを学習するクラス
    """
    def __init__(
        self,
        config: dict,
        target: str
    ) -> None:
        super().__init__(config, target)
        self._ridge_params = config.ridge_params

    def train_model(
        self,
        train_data: pd.DataFrame
    ):
        """学習を行う関数

        Args:
            train_data (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        # yを取り出す
        y_train = train_data[self._target].copy()
        X_train = train_data[self._feature_columns].copy()

        # 重回帰モデルを作成
        model = Ridge(alpha=self._ridge_params["alpha"])
        model.fit(X_train, y_train)

        # 切片と係数を取得
        coef = model.coef_
        intercept = model.intercept_
        return model, coef, intercept
