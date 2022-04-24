import numpy as np
import pandas as pd


class PredictModule:
    def __init__(
        self,
        config: dict,
        target: str,
        model_type: str,
        clip: bool = False
    ) -> None:
        """コンストラクタ

        Args:
            config (dict): parameter等を記載したconfigファイル
            model_type (str): lightgbm, xgboost, catboostから選択
        """
        self._primary_key = list(config.primary_key.keys())  # TypeError: unsupported operand type(s) for +: 'dict_keys' and 'list'
        self._feature_columns = list(config.feature_columns.keys())
        self._model_type = model_type
        self._target = target
        self._clip = clip

    def predict(self, model, test_data: pd.DataFrame) -> pd.DataFrame:
        # baseとなるdfの作成
        df_output = test_data[self._primary_key].copy()
        X_test = test_data[self._feature_columns].copy()

        # log
        print(f"pred {self._target} ...")
        # 予測
        y_pred = model.predict(X_test)
        # 予測値をdfに書き込む
        pred_col = f"pred_{self._target}"
        df_output[pred_col] = y_pred
        # clipするかどうか
        if self._clip:
            df_output[pred_col] = df_output[pred_col].clip(lower=0)  # 予測値の最小値は0にする
        return df_output
