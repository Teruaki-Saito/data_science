import logging
import inspect
import os
import pathlib
import yaml

from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler


@dataclass
class CreateLoggerBaseInfo:
    LOGGER_NAME = "LOGGER"
    DEFAULT_CONSOLE_LOG_LEVEL = "INFO"
    DEFAULT_FILE_LOG_LEVEL = "INFO"
    LOG_INFO = "%(asctime)s | %(levelname)s"
    LOG_MESSAGE = ("%(file_name)s#%(function_name)s() | line %(lineno)d | %(message)s")
    LOG_LEVEL_LIST = ["NONE", "DEBUG", "INFO", "WARNING", "ERROR"]

    # .envでsetされてなければ
    MODULE_NAME = os.getenv("MODULE_NAME") if os.getenv("MODULE_NAME") else "LOGGER"
    CURRENT_DIR = str(pathlib.Path(__file__).resolve().parent)

    # ロガーを作成
    log_base_dir = f"./logs/{MODULE_NAME}"
    os.makedirs(log_base_dir, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, "DEBUG"))
    log_text = f"{LOG_INFO} | {MODULE_NAME} | {LOG_MESSAGE}"
    log_format = logging.Formatter(log_text)

    # 環境変数設定ファイル取得
    env_variable = yaml.safe_load(
        open(f'{CURRENT_DIR}/../conf/log_configs.yaml', 'r', encoding="utf-8"))
    
    # log levelの設定
    CONSOLE_LOG_LEVEL = None
    FILE_LOG_LEVEL = None
    if CONSOLE_LOG_LEVEL:
        if CONSOLE_LOG_LEVEL in LOG_LEVEL_LIST:
            console_log_level_attr = getattr(logging, CONSOLE_LOG_LEVEL)
        else:
            console_log_level_attr = getattr(logging, DEFAULT_CONSOLE_LOG_LEVEL)
    else:
        console_log_level_attr = getattr(logging, env_variable['console_log_level'])

    if CONSOLE_LOG_LEVEL != 'NONE':
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level_attr)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        logging.getLogger(LOGGER_NAME).addHandler(console_handler)

    if FILE_LOG_LEVEL:
        if FILE_LOG_LEVEL in LOG_LEVEL_LIST:
            file_log_level_attr = getattr(logging, FILE_LOG_LEVEL)
        else:
            file_log_level_attr = getattr(logging, DEFAULT_FILE_LOG_LEVEL)
    else:
        file_log_level_attr = getattr(logging, env_variable['file_log_level'])

    if FILE_LOG_LEVEL != 'NONE':
        log_file_handler = TimedRotatingFileHandler(f'{log_base_dir}/module.log', encoding='utf-8', when='MIDNIGHT')
        log_file_handler.setLevel(file_log_level_attr)
        log_file_handler.setFormatter(log_format)
        logger.addHandler(log_file_handler)
        logging.getLogger(LOGGER_NAME).addHandler(log_file_handler)


def output_log(log_level, file_name, func_name, lineno, log_msg):
    '''
    ログを出力する

    Args：
        log_level(str): ログレベル
        file_name(str): 出力元ファイル名
        func_name(str): 出力元関数名
        lineno(int): 出力元行数
        log_msg(str): ログメッセージ
    Returns：
        None
    '''
    extra = {
        'file_name': file_name,
        'function_name': func_name,
        'lineno': lineno}

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.hasHandlers():
        create_logger()  # CreateLoggerBaseInfo
    if log_level == 'ERROR':
        logger.log(getattr(logging, log_level), log_msg, extra=extra, exc_info=True)
    else:
        logger.log(getattr(logging, log_level), log_msg, extra=extra)


def debug(log_msg):
    '''
    '''
    file_name = os.path.basename(inspect.currentframe().f_back.f_code.co_filename)
    func_name = inspect.currentframe().f_back.f_code.co_name
    lineno = inspect.currentframe().f_back.f_lineno
    output_log('DEBUG', file_name, func_name, lineno, log_msg)


def info(log_msg):
    '''
    '''
    file_name = os.path.basename(inspect.currentframe().f_back.f_code.co_filename)
    func_name = inspect.currentframe().f_back.f_code.co_name
    lineno = inspect.currentframe().f_back.f_lineno
    output_log('INFO', file_name, func_name, lineno, log_msg)
