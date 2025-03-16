import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from crypto_detector.config.settings import LOG_LEVEL, LOG_FILE, LOG_FORMAT


class Logger:
    """
    Клас для управління логуванням.
    Налаштовує логування у файл та консоль.
    """

    def __init__(self, name='crypto_detector', log_level=None, log_file=None, log_format=None):
        """
        Ініціалізація логера

        :param name: Назва логера
        :param log_level: Рівень логування (якщо None, використовується з налаштувань)
        :param log_file: Шлях до файлу логів (якщо None, використовується з налаштувань)
        :param log_format: Формат логів (якщо None, використовується з налаштувань)
        """
        self.name = name

        # Використання значень з налаштувань, якщо не вказані
        self.log_level = log_level or LOG_LEVEL
        self.log_file = log_file or LOG_FILE
        self.log_format = log_format or LOG_FORMAT

        # Створення директорії для логів, якщо вона не існує
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Створення логера
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """
        Налаштування логера

        :return: Налаштований логер
        """
        # Створення логера
        logger = logging.getLogger(self.name)
        logger.setLevel(self._get_log_level())

        # Очищення існуючих обробників
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)

        # Формат логів
        formatter = logging.Formatter(self.log_format)

        # Обробник для консолі
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Обробник для файлу (з ротацією логів)
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _get_log_level(self):
        """
        Перетворення рядкового рівня логування у відповідну константу

        :return: Константа рівня логування
        """
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        return levels.get(self.log_level.upper(), logging.INFO)

    def get_logger(self):
        """
        Отримання налаштованого логера

        :return: Логер
        """
        return self.logger


# Створення глобального логера
_logger = None


def get_logger(name='crypto_detector'):
    """
    Отримання глобального логера

    :param name: Назва логера
    :return: Логер
    """
    global _logger
    if _logger is None:
        _logger = Logger(name).get_logger()
    return _logger


def setup_monitoring_logger(symbol, directory=None):
    """
    Налаштування спеціального логера для моніторингу окремого символу

    :param symbol: Символ криптовалюти
    :param directory: Директорія для збереження логів
    :return: Логер
    """
    # Створення безпечної назви файлу
    safe_symbol = symbol.replace('/', '_')

    # Визначення директорії для логів
    if directory is None:
        directory = 'logs/monitoring'

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Створення унікальної назви файлу логів з датою
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(directory, f"{safe_symbol}_{date_str}.log")

    # Створення логера
    logger = Logger(
        name=f"monitoring_{safe_symbol}",
        log_file=log_file
    ).get_logger()

    return logger


def log_detection_result(result, logger=None):
    """
    Логування результату виявлення підозрілої активності

    :param result: Результат виявлення
    :param logger: Логер (якщо None, використовується глобальний логер)
    """
    if logger is None:
        logger = get_logger()

    symbol = result.get('symbol', 'Unknown')
    timestamp = result.get('timestamp', 'Unknown')
    probability = result.get('probability_score', 0)
    signals_count = len(result.get('signals', []))

    log_message = (
        f"Аналіз {symbol} на {timestamp}: "
        f"Ймовірність={probability:.4f}, Сигналів={signals_count}"
    )

    if probability > 0.7:
        logger.critical(log_message)
    elif probability > 0.5:
        logger.warning(log_message)
    elif probability > 0.3:
        logger.info(log_message)
    else:
        logger.debug(log_message)
