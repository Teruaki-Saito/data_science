from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import pandas as pd
import category_encoders as ce


def make_KFold(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    n_splits: int,
    random_state: int
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_idx = [(x, y) for (x, y) in kf.split(train_x, train_y)]
    return folds_idx


def make_stratifiedKFold(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    n_splits: int,
    random_state: int
):
    y_binned = pd.cut(train_y, bins=3, labels=None)
    y_binned = ce.OrdinalEncoder().fit_transform(y_binned.astype(str))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_idx = [(x, y) for (x, y) in skf.split(train_x, y_binned)]
    return folds_idx


def add_true_for_valid(
    df_val: pd.DataFrame,
    df_val_pred: pd.DataFrame,
    target: str,
    cv_num: int,
    config: dict
) -> pd.DataFrame:
    primary_key = list(config.primary_key.keys())
    _df = df_val[primary_key + [target]].copy()
    _df.columns = primary_key + [f"true_{target}"]
    _df["cv_num"] = cv_num
    df = pd.merge(df_val_pred, _df, how="left", on=primary_key + ["cv_num"])
    return df
