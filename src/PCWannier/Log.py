import logging

class Logger:
    _instance = None

    def __new__(cls, log_file: str = 'log.txt', level=logging.DEBUG):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(log_file, level)
        return cls._instance

    def __init__(self, log_file: str, level=logging.DEBUG):
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(level)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    @staticmethod
    def info(message: str):
        Logger._instance.logger.info(message)

    @staticmethod
    def warning(message: str):
        Logger._instance.logger.warning(message)

    @staticmethod
    def error(message: str, exc_info=True):
        Logger._instance.logger.error(message, exc_info=exc_info)

    @staticmethod
    def debug(message: str):
        Logger._instance.logger.debug(message)