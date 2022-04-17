from common_module.utils.base_config_manager import BaseConfigManager


class ConfigManager(BaseConfigManager):
    """
    カットの学習、シミュ用の予測で使う設定情報を格納するクラス

    Args:
        ConfigManager (config_manager): ベースとなるconfig_manager
    """
    def __init__(self, config_path):
        super().__init__(config_path)
