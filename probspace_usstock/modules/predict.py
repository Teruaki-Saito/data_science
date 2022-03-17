import os
import gc
import datetime
import pickle

import numpy as np
import pandas as pd


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
