import os
import pandas as pd
import pandas_profiling as pdp
import sweetviz as sv


def get_pandas_profiling(
    df: pd.DataFrame,
    output_path: str,
    output_name: str
) -> None:
    """pandas profilingによるサマリ

    Args:
        df (pd.DataFrame): _description_
        output_path (str): _description_
        output_name (str): _description_
    """
    profile = pdp.ProfileReport(df)
    os.makedirs(output_path, exist_ok=True)
    profile.to_file(os.path.join(output_path, output_name) + ".html")


def get_sweetviz_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_path: str,
    output_name: str
) -> None:
    """sweetvizによるサマリ。trainとtestのデータを比較する

    Args:
        train (pd.DataFrame): _description_
        test (pd.DataFrame): _description_
        output_path (str): _description_
        output_name (str): _description_
    """
    report = sv.compare([train, "Train"], [test, "Test"])
    os.makedirs(output_path, exist_ok=True)
    report.show_html(os.path.join(output_path, output_name) + ".html")
