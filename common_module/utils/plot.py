from turtle import title
import pandas as pd
import numpy as np
import os
import sys
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import ptitprince
from pygam import LinearGAM, s
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'


def plot_venn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str
) -> None:
    """trainとtestのベン図を描画する

    Args:
        train (pd.DataFrame): trainデータ
        test (pd.DataFrame): testデータ
        target_col (str): 描画したい列
    """
    print(f"■ {target_col}")
    train_unique = train[target_col].unique()
    test_unique = test[target_col].unique()
    common_num = len(set(train_unique) & set(test_unique))
    
    # ベン図のプロット
    plt.figure(figsize=(2, 2))
    venn2(
        subsets=(
            len(train_unique) - common_num,
            len(test_unique) - common_num,
            common_num
        ),
        set_labels=("Train", "Test")
    )
    plt.show()
    
    # 各々にしかないデータ
    train_only = set(train[target_col]) - set(test[target_col])
    test_only = set(test[target_col]) - set(train[target_col])
    print(f"Only Train exist {train_only}"[:100] + "....")
    print(f"Only Test exist {test_only}"[:100] + "....")


def plot_raincloud(
    df: pd.DataFrame,
    target_cols: List[str],
    figsize: Tuple[int] = (12, 6)
) -> None:
    """raincloudを描画する関数
    rain cloud plot とは violin-plot と box-plot を同時に描いて、分布の様子と統計量を確認しやすく工夫した可視化方法のことです。
    連続値の分布とどうなっているか直感的にわかりますし、具体的な値がどうなっているかの確認も同時に行えるので便利です。

    Args:
        df (pd.DataFrame): _description_
        target_cols (List[str]): _description_
        figsize (Tuple[int], optional): _description_. Defaults to (12, 6).
    """
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ptitprince.RainCloud(data=df[target_cols], ax=ax, orient='v')


def plot_histogram(
    df: pd.DataFrame,
    target_cols: List[str],
    ncols: int
) -> None:
    """目的列の分布を描画する関数

    Args:
        df (pd.DataFrame): _description_
        target_cols (List[str]): _description_
        ncols (int): _description_
    """
    nrows = (len(target_cols) + 1) // ncols
    fig, axes = plt.subplots(
        figsize=(4 * ncols, 4 * nrows),
        nrows=nrows,
        ncols=ncols
    )
    for target_col, ax in zip(target_cols, np.ravel(axes)):
        sns.histplot(df[target_col], ax=ax, label=target_col)
        ax.set_xlabel('')
        ax.legend()
        ax.grid()
    fig.tight_layout()


def plot_histgram_for_valid(
    data_true: pd.Series,
    data_pred: pd.Series,
    title: str = "target | cv_num",
    figsize: Tuple[int, int] = (4, 4),
    alpha: float = 0.5,
    bins: int = 20
) -> None:
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    # グラフの描画
    ax.hist(data_true, alpha=alpha, bins=bins, label="true", color="tab:blue")
    ax.hist(data_pred, alpha=alpha, bins=bins, label="pred", color="tab:orange")
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_heatmap(
    df: pd.DataFrame,
    target_cols: List[str],
    figsize: Tuple[int] = (10, 10)
) -> None:
    """Targetどうしの相関を描画する関数

    Args:
        df (pd.DataFrame): _description_
        target_cols (List[str]): _description_
        figsize (Tuple[int], optional): _description_. Defaults to (12, 6).
    """
    corr = df[target_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax, cmap='Blues')


def plot_lineargam(
    df: pd.DataFrame,
    target_col: str,
    comparison_cols: List[str],
    figsize: Tuple[int] = (14, 5)
) -> None:
    """GAMによる
    GAM (一般的加法モデル: Generatized Additive Model) は予測に対して、特徴量ごとに独立した影響を考えるシンプルなモデルです。
    特徴量ごとに独立して考えるモデルゆえ「ある特徴量が大きくなった時予測する対象の値はどう変化するか」を見積もることができます。
    (ただし特徴量ごとにしか考えれないため、交互作用については考慮することができません. 
    例えば 2000 年の RPG だけ得意に売れた、というように複数の特徴が同時に特定の値になった時の影響を考慮できません)
    【使い所】
    ・入力と予測したい変数の関係性をわかりやすく知れる為、因果関係を調べる上での手がかりになる
    ・実務的には予測性能というより、ある変数が予測したい値とどういうふうに関連があるかが知りたい場合

    Args:
        df (pd.DataFrame): _description_
        target_col (str): _description_
        comparison_cols (List[str]): _description_
        figsize (Tuple[int], optional): _description_. Defaults to (14, 5).
    """
    # comparison_colsにnullがある場合に中央値で補完する
    df_comp = df[comparison_cols].copy()
    df_comp = df_comp.fillna(df_comp.median())

    # targetの情報を取得する
    y = df[target_col]
    print("■", target_col)

    # gam学習
    model = LinearGAM()
    model.fit(df_comp, y)

    # plot
    fig, axes = plt.subplots(
        ncols=len(df_comp.columns),
        figsize=figsize,
        sharey=True,
        tight_layout=True
    )
    axes = np.array(axes).flatten()
    for i, (ax, title, p_value) in enumerate(zip(axes, df_comp.columns, model.statistics_["p_values"])):
        XX = model.generate_X_grid(term=i)
        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
        ax.axhline(0, c='#cccccc')
        ax.set_title("{0:} (p={1:.2})".format(title, p_value))
        ax.set_yticks([])
        ax.grid()
    fig.tight_layout()


def plot_feature_importance_for_valid(importance_set, target, n_splits):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.
    """
    df = pd.DataFrame()
    for cv_num in range(n_splits):
        _df = importance_set[f"{target}_{cv_num}"].copy()
        _df["cv_num"] = cv_num
        df = pd.concat([df, _df], axis=0, ignore_index=True)
    
    order = df.groupby("feature").sum()[["gain"]].sort_values("gain", ascending=False).index

    fig, ax = plt.subplots(figsize=(10, max(6, len(order) * .2)))
    sns.boxenplot(data=df, y="feature", x="gain", order=order, ax=ax, palette='viridis', orient='h')
    ax.set_title(f"feature importances | {target}")
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    fig.tight_layout()