import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib


def parse_timeframe(timeframe):
    """
    Переведення часового інтервалу у хвилини

    :param timeframe: Рядок з часовим інтервалом ('1m', '5m', '1h', '1d', тощо)
    :return: Кількість хвилин
    """
    if not isinstance(timeframe, str):
        raise ValueError("Timeframe повинен бути рядком")

    unit = timeframe[-1].lower()
    try:
        value = int(timeframe[:-1])
    except ValueError:
        raise ValueError(f"Невірний формат timeframe: {timeframe}")

    if unit == 'm':  # Хвилини
        return value
    elif unit == 'h':  # Години
        return value * 60
    elif unit == 'd':  # Дні
        return value * 24 * 60
    elif unit == 'w':  # Тижні
        return value * 7 * 24 * 60
    else:
        raise ValueError(f"Непідтримувана одиниця часу: {unit}")


def format_timestamp(timestamp, format_str='%Y-%m-%d %H:%M:%S'):
    """
    Форматування часової мітки

    :param timestamp: Часова мітка (datetime, int, float або str)
    :param format_str: Формат для виведення
    :return: Форматований рядок
    """
    if isinstance(timestamp, datetime):
        return timestamp.strftime(format_str)
    elif isinstance(timestamp, (int, float)):
        # Перевірка, чи це мілісекунди або секунди
        if timestamp > 1e11:  # Швидше за все, це мілісекунди
            return datetime.fromtimestamp(timestamp / 1000).strftime(format_str)
        else:
            return datetime.fromtimestamp(timestamp).strftime(format_str)
    elif isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime(format_str)
        except ValueError:
            return timestamp
    else:
        return str(timestamp)


def extract_base_quote(symbol):
    """
    Розділення символу на базову та котирувальну валюти

    :param symbol: Символ криптовалюти (наприклад, 'BTC/USDT')
    :return: Кортеж (базова валюта, котирувальна валюта)
    """
    if '/' in symbol:
        base, quote = symbol.split('/')
        return base, quote
    else:
        # Спроба угадати розділення для символів без '/'
        for quote_currency in ['USDT', 'USD', 'BTC', 'ETH', 'BUSD', 'USDC']:
            if symbol.endswith(quote_currency):
                base = symbol[:-len(quote_currency)]
                return base, quote_currency

        # Якщо не вдалося визначити, повертаємо весь символ як базову валюту
        return symbol, ''


def generate_filename(prefix, symbol, start_date=None, end_date=None, extension='csv'):
    """
    Генерація імені файлу для збереження даних

    :param prefix: Префікс файлу
    :param symbol: Символ криптовалюти
    :param start_date: Початкова дата (опціонально)
    :param end_date: Кінцева дата (опціонально)
    :param extension: Розширення файлу
    :return: Ім'я файлу
    """
    # Безпечна версія символу для використання у імені файлу
    safe_symbol = symbol.replace('/', '_')

    # Формування частини імені файлу з датами
    date_part = ''
    if start_date:
        start_str = start_date if isinstance(start_date, str) else start_date.strftime('%Y%m%d')
        date_part += f"_{start_str}"

        if end_date:
            end_str = end_date if isinstance(end_date, str) else end_date.strftime('%Y%m%d')
            date_part += f"_{end_str}"
    else:
        # Якщо дати не вказано, використовуємо поточну дату
        date_part = f"_{datetime.now().strftime('%Y%m%d')}"

    # Ім'я файлу
    filename = f"{prefix}_{safe_symbol}{date_part}.{extension}"

    return filename


def load_json_file(file_path, default=None):
    """
    Завантаження даних з JSON-файлу

    :param file_path: Шлях до файлу
    :param default: Значення за замовчуванням, якщо файл не існує або виникла помилка
    :return: Дані з файлу або значення за замовчуванням
    """
    if not os.path.exists(file_path):
        return default

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Помилка при завантаженні файлу {file_path}: {str(e)}")
        return default


def save_json_file(data, file_path, indent=2):
    """
    Збереження даних у JSON-файл

    :param data: Дані для збереження
    :param file_path: Шлях до файлу
    :param indent: Відступ для форматування JSON
    :return: True, якщо збереження успішне, інакше False
    """
    try:
        # Створення директорії, якщо вона не існує
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=json_serializer)

        return True
    except Exception as e:
        print(f"Помилка при збереженні файлу {file_path}: {str(e)}")
        return False


def json_serializer(obj):
    """
    Допоміжна функція для серіалізації об'єктів у JSON

    :param obj: Об'єкт для серіалізації
    :return: Серіалізований об'єкт
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def calculate_date_range(end_date=None, days_back=30):
    """
    Розрахунок діапазону дат

    :param end_date: Кінцева дата (якщо None, використовується поточна дата)
    :param days_back: Кількість днів назад для початкової дати
    :return: Кортеж (початкова дата, кінцева дата) у форматі 'YYYY-MM-DD'
    """
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    start_date = end_date - timedelta(days=days_back)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def generate_cache_key(symbol, timeframe, start_date, end_date):
    """
    Генерація ключа для кешування даних

    :param symbol: Символ криптовалюти
    :param timeframe: Часовий інтервал
    :param start_date: Початкова дата
    :param end_date: Кінцева дата
    :return: Унікальний ключ
    """
    # Нормалізація параметрів
    symbol = str(symbol).upper()
    timeframe = str(timeframe).lower()
    start_date = format_timestamp(start_date, '%Y%m%d')
    end_date = format_timestamp(end_date, '%Y%m%d')

    # Створення строки для хешування
    key_string = f"{symbol}_{timeframe}_{start_date}_{end_date}"

    # Хешування для отримання короткого унікального ключа
    hash_object = hashlib.md5(key_string.encode())
    return hash_object.hexdigest()


def calculate_change_percentage(current, previous):
    """
    Розрахунок відсотка зміни між значеннями

    :param current: Поточне значення
    :param previous: Попереднє значення
    :return: Відсоток зміни
    """
    if previous == 0:
        return float('inf') if current > 0 else 0

    return ((current / previous) - 1) * 100
