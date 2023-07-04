import logging
def prepare_logger() -> logging.Logger:
    # ロガーの作成
    logger = logging.getLogger(__name__)
    # ログハンドラの作成
    handler = logging.StreamHandler()
    # ログフォーマットの設定
    formatter = logging.Formatter('xxxxxxxxxxxxx : %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # ログハンドラをロガーに追加
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger