import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error
)


def calc_MAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """平均絶対誤差 (MAE, Mean Absolute Error)を計算する関数
    ■参考：Pythonでデータサイエンス
    平均絶対誤差 (MAE, Mean Absolute Error) は、実際の値と予測値の絶対値を平均したものです。
    MAE が小さいほど誤差が少なく、予測モデルが正確に予測できていることを示し、MAE が大きいほど実際の値と予測値
    に誤差が大きく、予測モデルが正確に予測できていないといえます。
    ■Kaggle本
    外れ値の影響を低減した形での評価に適した関数
    ■その他
    MAEは[実測値、予測値]の時、[10000, 9500]と[1000, 500]をそれぞれ500の誤差として等しく扱う。
    前者は誤差だが後者は致命的。実測値と予測値の差を最小化するのではなく、実測値と予測値の比を1に近づけるように
    最適化する方が実用的。よってこのような場合は対数変換を噛ませるのが良い。


    Args:
        y_true (pd.Series): _description_
        y_pred (pd.Series): _description_

    Returns:
        float: _description_
    """
    MAE = mean_absolute_error(y_true, y_pred)
    return MAE


def calc_MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """平均二乗誤差 (MSE, Mean Squared Error) を計算する関数
    平均二乗誤差 (MSE, Mean Squared Error) とは、実際の値と予測値の絶対値の 2 乗を平均したものです
    このため、MAE に比べて大きな誤差が存在するケースで、大きな値を示す特徴があります。

    Args:
        y_true (pd.Series): _description_
        y_pred (pd.Series): _description_

    Returns:
        float: _description_
    """
    MSE = mean_squared_error(y_true, y_pred)
    return MSE


def calc_RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """二乗平均平方根誤差 (RMSE: Root Mean Squared Error) 
    MSEを二乗したことの影響を平方根で補正したものです。
    ■RMSEのポイント
    MAEと比較すると外れ値の影響を受けやすいので、あらかじめ外れ値を除く処理などをしておかないと
    外れ値に過剰に適合したモデルを作成してしまう可能性がある。

    Args:
        y_true (pd.Series): _description_
        y_pred (pd.Series): _description_

    Returns:
        float: _description_
    """
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    return RMSE


def calc_RMSLE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """RMSLE(Root Mean Squared Logarithmic Error)を計算する関数
    真の値と予測値の対数をそれぞれ撮った後のさの2条の平均の平方根によって計算される指標。
    目的変数の対数をとって変換した値を新たな目的変数とした上でRMSEを最小化すればRMSLEを最小化することになる。
    目的変数が裾の思い分布をもち、変換しないままだと大きな値の影響が強い場合や、真の値と予測値の比率に着目したい
    場合に用いられる。
    対数を取るにあたっては、真の値が0の時に値が負に発散するのを避けるため、
    通常は1を加えてから対数をとる。numpyのlog1pが使用できる。

    Args:
        y_true (pd.Series): _description_
        y_pred (pd.Series): _description_

    Returns:
        float: _description_
    """
    RMSLE = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return RMSLE