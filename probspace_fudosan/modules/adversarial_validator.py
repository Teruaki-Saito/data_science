from os import stat
import pandas as pd
import numpy as np
import lightgbm as lgb
from probspace_fudosan.modules.validator import (
    make_KFold,
    make_stratifiedKFold
)
from probspace_fudosan.modules.train_module import TrainLGBModuleAdv
from probspace_fudosan.modules.predict import PredictModule


class AdversarialValidator:
    @staticmethod
    def create_AdvVal_target_column(
        df_all: pd.DataFrame,
        config: dict
    ) -> pd.DataFrame:
        df = df_all.copy()
        # trainは0, testは1とする
        df.loc[df["data_type"] == "train", config.AdversarialValidation["target"]] = 0
        df.loc[df["data_type"] == "test", config.AdversarialValidation["target"]] = 1
        return df

    @staticmethod
    def exec_cross_validation(
        df_all: pd.DataFrame,
        cv_method: str,
        config: dict
    ) -> pd.DataFrame:
        df = df_all.copy()

        # 変数の設定
        CONFIG = config.AdversarialValidation
        TARGET = CONFIG["target"]

        # cv インスタンスの判定
        if cv_method == "KFold":
            folds_idx = make_KFold(
                train_x=df,
                train_y=df[TARGET],
                n_splits=config.n_splits,
                random_state=config.random_state
            )
        elif cv_method == "StratifiedKFold":
            folds_idx = make_stratifiedKFold(
                train_x=df,
                train_y=df[TARGET],
                n_splits=config.n_splits,
                random_state=config.random_state
            )

        # 学習用インスタンスの作成
        ins_train = TrainLGBModuleAdv(
            config=config,
            target=TARGET
        )
        # 予測用インスタンスの作成
        ins_pred = PredictModule(config=config, target=TARGET, model_type="lightgbm", is_test=False, clip=False)

        # 学習初期設定
        model_set = {}
        importance_set = {}
        evals_set = {}
        df_val_list = []
        # 学習
        for cv_num, (tr_idx, va_idx) in enumerate(folds_idx):

            print(f"start cv_num={cv_num} ...")

            df_tr_tr, df_tr_va = df.loc[tr_idx, :], df.loc[va_idx, :]
            model, df_imp, evals_result = ins_train.train_model(
                df_train=df_tr_tr,
                df_val=df_tr_va,
                have_weight=False
            )
            # dataset への保存
            model_set[f"{TARGET}_{cv_num}"] = model
            importance_set[f"{TARGET}_{cv_num}"] = df_imp
            evals_set[f"{TARGET}_{cv_num}"] = evals_result
            # 予測
            df_pred_cv = ins_pred.predict(model=model, test_df=df_tr_va, cv_num=cv_num)
            df_val_list.append(df_pred_cv)

        # 結果をconcat
        df_val_all = pd.concat(df_val_list, ignore_index=True, sort=False)
        return model_set, importance_set, evals_set, df_val_all

    @staticmethod
    def splilt_testLike_data(df_val_all: pd.DataFrame, test_size: float, config: dict):
        df = df_val_all.copy()
        TARGET = config.AdversarialValidation["target"]

        test_len = np.ceil(len(df) * test_size)

        df = df.sort_values(TARGET, ascending=False)  # テストデータっぽいデータが上位にそーとされる
        df_train = df.loc[test_len:, :]
        df_val = df.loc[:test_len, :]
        print(f"df_train: {df_train.shape}")
        print(f"df_val: {df_val.shape}")
        return df_train, df_val
