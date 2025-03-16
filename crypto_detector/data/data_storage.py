import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


class DataStorage:
    """
    Клас для зберігання та завантаження даних.
    Відповідає за серіалізацію/десеріалізацію різних типів даних та управління файлами.
    """

    def __init__(self, base_dir='data'):
        """
        Ініціалізація системи зберігання даних

        :param base_dir: Базова директорія для зберігання даних
        """
        self.base_dir = base_dir

        # Створення структури директорій для різних типів даних
        self.data_dirs = {
            'ohlcv': os.path.join(base_dir, 'ohlcv'),
            'orderbook': os.path.join(base_dir, 'orderbook'),
            'models': os.path.join(base_dir, 'models'),
            'weights': os.path.join(base_dir, 'weights'),
            'results': os.path.join(base_dir, 'results'),
            'alerts': os.path.join(base_dir, 'alerts'),
            'latest': os.path.join(base_dir, 'latest')
        }

        # Створення директорій, якщо вони не існують
        for dir_path in self.data_dirs.values():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def get_data_path(self, symbol, timeframe, start_date, end_date, is_latest=False):
        """
        Формування шляху до файлу з даними

        :param symbol: Символ криптовалюти
        :param timeframe: Часовий інтервал
        :param start_date: Початкова дата
        :param end_date: Кінцева дата
        :param is_latest: Чи є це останніми даними
        :return: Шлях до файлу
        """
        # Нормалізація символу для використання у назві файлу
        symbol_safe = symbol.replace('/', '_')

        # Форматування дат
        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y%m%d')
        else:
            start_str = start_date

        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y%m%d')
        else:
            end_str = end_date

        # Визначення директорії для зберігання
        if is_latest:
            data_dir = self.data_dirs['latest']
            filename = f"{symbol_safe}_{timeframe}_latest.csv"
        else:
            data_dir = self.data_dirs['ohlcv']
            filename = f"{symbol_safe}_{timeframe}_{start_str}_{end_str}.csv"

        # Створення директорії для символу, якщо вона не існує
        symbol_dir = os.path.join(data_dir, symbol_safe)
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)

        return os.path.join(symbol_dir, filename)

    def save_data(self, data, symbol, timeframe, start_date, end_date, is_latest=False):
        """
        Збереження даних у файл

        :param data: DataFrame з даними
        :param symbol: Символ криптовалюти
        :param timeframe: Часовий інтервал
        :param start_date: Початкова дата
        :param end_date: Кінцева дата
        :param is_latest: Чи є це останніми даними
        :return: Шлях до збереженого файлу
        """
        if data is None or data.empty:
            print(f"Немає даних для збереження для {symbol}")
            return None

        file_path = self.get_data_path(symbol, timeframe, start_date, end_date, is_latest)

        try:
            # Збереження даних у файл
            data.to_csv(file_path)
            print(f"Дані успішно збережено у {file_path}")
            return file_path
        except Exception as e:
            print(f"Помилка при збереженні даних у {file_path}: {e}")
            return None

    def load_data(self, file_path):
        """
        Завантаження даних з файлу

        :param file_path: Шлях до файлу з даними
        :return: DataFrame з даними
        """
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не існує")
            return pd.DataFrame()

        try:
            # Завантаження даних з файлу
            data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            print(f"Дані успішно завантажено з {file_path}")
            return data
        except Exception as e:
            print(f"Помилка при завантаженні даних з {file_path}: {e}")
            return pd.DataFrame()

    def save_orderbook(self, orderbook, symbol, timestamp=None):
        """
        Збереження книги ордерів

        :param orderbook: Дані книги ордерів
        :param symbol: Символ криптовалюти
        :param timestamp: Часова мітка (за замовчуванням - поточний час)
        :return: Шлях до збереженого файлу
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Нормалізація символу для використання у назві файлу
        symbol_safe = symbol.replace('/', '_')

        # Форматування часової мітки
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')

        # Створення директорії для символу, якщо вона не існує
        symbol_dir = os.path.join(self.data_dirs['orderbook'], symbol_safe)
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)

        # Шлях до файлу
        file_path = os.path.join(symbol_dir, f"orderbook_{symbol_safe}_{timestamp_str}.json")

        try:
            # Додавання часової мітки до даних
            orderbook_data = orderbook.copy()
            orderbook_data['timestamp'] = timestamp.isoformat()

            # Збереження даних у файл
            with open(file_path, 'w') as f:
                json.dump(orderbook_data, f, indent=2)

            print(f"Книгу ордерів успішно збережено у {file_path}")
            return file_path
        except Exception as e:
            print(f"Помилка при збереженні книги ордерів у {file_path}: {e}")
            return None

    def load_orderbook(self, file_path):
        """
        Завантаження книги ордерів з файлу

        :param file_path: Шлях до файлу з книгою ордерів
        :return: Дані книги ордерів
        """
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не існує")
            return None

        try:
            # Завантаження даних з файлу
            with open(file_path, 'r') as f:
                orderbook = json.load(f)

            print(f"Книгу ордерів успішно завантажено з {file_path}")
            return orderbook
        except Exception as e:
            print(f"Помилка при завантаженні книги ордерів з {file_path}: {e}")
            return None

    def save_alert(self, alert_data, symbol, timestamp=None):
        """
        Збереження даних сповіщення

        :param alert_data: Дані сповіщення
        :param symbol: Символ криптовалюти
        :param timestamp: Часова мітка (за замовчуванням - поточний час)
        :return: Шлях до збереженого файлу
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Нормалізація символу для використання у назві файлу
        symbol_safe = symbol.replace('/', '_')

        # Форматування часової мітки
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')

        # Шлях до файлу
        file_path = os.path.join(self.data_dirs['alerts'], f"alert_{symbol_safe}_{timestamp_str}.json")

        try:
            # Збереження даних у файл
            with open(file_path, 'w') as f:
                json.dump(alert_data, f, indent=2, default=self._json_serializer)

            print(f"Дані сповіщення успішно збережено у {file_path}")
            return file_path
        except Exception as e:
            print(f"Помилка при збереженні даних сповіщення у {file_path}: {e}")
            return None

    def save_backtest_results(self, results, symbol, start_date, end_date, is_adaptive=False):
        """
        Збереження результатів бектестингу

        :param results: Результати бектестингу
        :param symbol: Символ криптовалюти
        :param start_date: Початкова дата
        :param end_date: Кінцева дата
        :param is_adaptive: Чи є це результатами адаптивного бектестингу
        :return: Шлях до збереженого файлу
        """
        # Нормалізація символу для використання у назві файлу
        symbol_safe = symbol.replace('/', '_')

        # Форматування дат
        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y%m%d')
        else:
            start_str = start_date

        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y%m%d')
        else:
            end_str = end_date

        # Визначення префіксу в залежності від типу бектестингу
        prefix = "adaptive_" if is_adaptive else "standard_"

        # Шлях до файлу
        file_path = os.path.join(self.data_dirs['results'],
                                 f"{prefix}backtest_{symbol_safe}_{start_str}_{end_str}.json")

        try:
            # Збереження даних у файл
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=self._json_serializer)

            print(f"Результати бектестингу успішно збережено у {file_path}")
            return file_path
        except Exception as e:
            print(f"Помилка при збереженні результатів бектестингу у {file_path}: {e}")
            return None

    def save_model(self, model, model_name):
        """
        Збереження ML моделі

        :param model: ML модель
        :param model_name: Назва моделі
        :return: Шлях до збереженої моделі
        """
        # Шлях до файлу
        file_path = os.path.join(self.data_dirs['models'], f"{model_name}.pkl")

        try:
            # Збереження моделі у файл
            import joblib
            joblib.dump(model, file_path)

            print(f"Модель успішно збережено у {file_path}")
            return file_path
        except Exception as e:
            print(f"Помилка при збереженні моделі у {file_path}: {e}")
            return None

    def load_model(self, model_name):
        """
        Завантаження ML моделі

        :param model_name: Назва моделі
        :return: Завантажена модель
        """
        # Шлях до файлу
        file_path = os.path.join(self.data_dirs['models'], f"{model_name}.pkl")

        if not os.path.exists(file_path):
            print(f"Файл моделі {file_path} не існує")
            return None

        try:
            # Завантаження моделі з файлу
            import joblib
            model = joblib.load(file_path)

            print(f"Модель успішно завантажено з {file_path}")
            return model
        except Exception as e:
            print(f"Помилка при завантаженні моделі з {file_path}: {e}")
            return None

    def save_signal_weights(self, weights, category=None):
        """
        Збереження ваг сигналів

        :param weights: Словник з вагами сигналів
        :param category: Категорія токена (якщо None, зберігаються загальні ваги)
        :return: Шлях до збереженого файлу
        """
        # Визначення імені файлу
        filename = f"{category}_weights.json" if category else "signal_weights.json"

        # Шлях до файлу
        file_path = os.path.join(self.data_dirs['weights'], filename)

        try:
            # Збереження ваг у файл
            with open(file_path, 'w') as f:
                json.dump(weights, f, indent=2)

            print(f"Ваги сигналів успішно збережено у {file_path}")
            return file_path
        except Exception as e:
            print(f"Помилка при збереженні ваг сигналів у {file_path}: {e}")
            return None

    def load_signal_weights(self, category=None):
        """
        Завантаження ваг сигналів

        :param category: Категорія токена (якщо None, завантажуються загальні ваги)
        :return: Словник з вагами сигналів
        """
        # Визначення імені файлу
        filename = f"{category}_weights.json" if category else "signal_weights.json"

        # Шлях до файлу
        file_path = os.path.join(self.data_dirs['weights'], filename)

        if not os.path.exists(file_path):
            print(f"Файл ваг сигналів {file_path} не існує")
            return {}

        try:
            # Завантаження ваг з файлу
            with open(file_path, 'r') as f:
                weights = json.load(f)

            print(f"Ваги сигналів успішно завантажено з {file_path}")
            return weights
        except Exception as e:
            print(f"Помилка при завантаженні ваг сигналів з {file_path}: {e}")
            return {}

    def _json_serializer(self, obj):
        """
        Допоміжна функція для серіалізації об'єктів у JSON

        :param obj: Об'єкт для серіалізації
        :return: Серіалізований об'єкт
        """
        # Обробка типів, які не підтримуються напряму в JSON
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
