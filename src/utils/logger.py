import logging
import os
from logging.handlers import RotatingFileHandler
from config.paths import LOGS_PATH


class ModeLogger:
    def __init__(
        self,
        mode,
        log_dir=LOGS_PATH,
        log_to_console=False,
        max_log_size=5 * 1024 * 1024,
        backup_count=5,
    ):
        self.mode = mode.lower()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(self.mode)
        self.logger.setLevel(logging.DEBUG)

        log_file = os.path.join(log_dir, f"{self.mode}.log")

        if not self.logger.hasHandlers():
            try:
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=max_log_size, backupCount=backup_count
                )
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

                if log_to_console:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
            except Exception as e:
                self.logger.error(f"Error initializing logger for {self.mode}: {e}")
                raise

    def get_logger(self):
        return self.logger


preprocessing_logger = ModeLogger(
    "data_preprocessing", log_to_console=False
).get_logger()
training_logger = ModeLogger("model_training", log_to_console=False).get_logger()
inference_logger = ModeLogger("inference", log_to_console=False).get_logger()
