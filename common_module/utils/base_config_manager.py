from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict
from abc import ABCMeta
import yaml


class BaseConfigManager(metaclass=ABCMeta):
    """
    設定情報を格納するデータクラス

    Args:
        config_path(str): 設定情報の書かれたyamlファイルのパス
    """
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for config_key, config_value in config.items():
            setattr(self, config_key, config_value)
