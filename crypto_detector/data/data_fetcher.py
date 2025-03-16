import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta

from crypto_detector.data.exchange_client import ExchangeClient
from crypto_detector.data.data_storage import DataStorage


class DataFetcher:
    """
    Клас для завантаження та підготовки даних з криптовалютних бірж.
    Відповідає за отримання історичних даних, агрегацію та підготовку даних для аналізу.
    """

    def __init__(self, exchange_client=None, storage=None, default_timeframe='5m'):
        """
        Ініціалізація завантажувача даних

        :param exchange_client: Клієнт для взаємодії з біржею
        :param storage: Система зберігання даних
        :param default_timeframe: Часовий інтервал за замовчуванням
        """
        self.exchange_client = exchange_client or ExchangeClient()
        self.storage = storage or DataStorage()
        self.default_timeframe = default_timeframe

    async def fetch_historical_data(self, symbol, start_date, end_date, timeframe=None):
        """
        Завантаження історичних даних за вказаний період

        :param symbol: Символ криптовалюти
        :param start_date: Початкова дата (рядок 'YYYY-MM-DD' або об'єкт datetime)
        :param end_date: Кінцева дата (рядок 'YYYY-MM-DD' або об'єкт datetime)
        :param timeframe: Часовий інтервал (за замовчуванням використовується self.default_timeframe)
        :return: DataFrame з історичними даними або шлях до збереженого файлу
        """
        timeframe = timeframe or self.default_timeframe

        # Конвертація дат, якщо вони передані як рядки
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Перевірка, чи є вже завантажені дані
        stored_data_path = self.storage.get_data_path(symbol, timeframe, start_date, end_date)
        if os.path.exists(stored_data_path):
            print(
                f"Знайдено збережені дані для {symbol} з {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
            return self.storage.load_data(stored_data_path)

        # Завантаження даних по частинах, щоб уникнути обмежень API
        all_data = []
        current_date = start_date

        while current_date <= end_date:
            next_date = min(current_date + timedelta(days=7), end_date)

            # Конвертація timestamp в мілісекунди для CCXT
            since = int(current_date.timestamp() * 1000)

            try:
                # Завантаження даних
                ohlcv_data = await self.exchange_client.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=1000  # Максимальна кількість свічок
                )

                if ohlcv_data is not None and not ohlcv_data.empty:
                    all_data.append(ohlcv_data)
                    print(
                        f"Завантажено дані для {symbol} з {current_date.strftime('%Y-%m-%d')} по {next_date.strftime('%Y-%m-%d')}")
                else:
                    print(
                        f"Немає даних для {symbol} з {current_date.strftime('%Y-%m-%d')} по {next_date.strftime('%Y-%m-%d')}")

                # Затримка, щоб не перевищити ліміти API
                await asyncio.sleep(1)

            except Exception as e:
                print(
                    f"Помилка при завантаженні даних для {symbol} з {current_date.strftime('%Y-%m-%d')} по {next_date.strftime('%Y-%m-%d')}: {e}")

            current_date = next_date + timedelta(seconds=1)  # Додаємо 1 секунду, щоб уникнути дублювання

        # Об'єднання всіх завантажених даних
        if all_data:
            combined_data = pd.concat(all_data)
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  # Видалення дублікатів
            combined_data.sort_index(inplace=True)

            # Збереження даних
            self.storage.save_data(combined_data, symbol, timeframe, start_date, end_date)

            return combined_data
        else:
            print(
                f"Не вдалося завантажити дані для {symbol} з {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()

    async def fetch_latest_data(self, symbol, lookback_period=24, timeframe=None):
        """
        Завантаження останніх даних для аналізу

        :param symbol: Символ криптовалюти
        :param lookback_period: Період в годинах для аналізу
        :param timeframe: Часовий інтервал (за замовчуванням використовується self.default_timeframe)
        :return: DataFrame з останніми даними
        """
        timeframe = timeframe or self.default_timeframe

        # Розрахунок кількості свічок на основі lookback_period
        minutes_in_timeframe = self.exchange_client.convert_timeframe_to_minutes(timeframe)
        limit = int((lookback_period * 60) / minutes_in_timeframe) + 10  # +10 для запасу

        try:
            # Завантаження останніх даних
            ohlcv_data = await self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=limit)

            if ohlcv_data is not None and not ohlcv_data.empty:
                # Зберігання останніх даних для можливого використання в майбутньому
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=lookback_period)
                self.storage.save_data(ohlcv_data, symbol, timeframe, start_date, end_date, is_latest=True)

                return ohlcv_data
            else:
                print(f"Не вдалося завантажити останні дані для {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Помилка при завантаженні останніх даних для {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_orderbook_data(self, symbol, limit=20):
        """
        Завантаження даних книги ордерів

        :param symbol: Символ криптовалюти
        :param limit: Глибина книги ордерів
        :return: Дані книги ордерів
        """
        try:
            order_book = await self.exchange_client.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            print(f"Помилка при завантаженні книги ордерів для {symbol}: {e}")
            return {'bids': [], 'asks': []}

    async def fetch_available_symbols(self, quote_currency='USDT'):
        """
        Отримання списку доступних символів

        :param quote_currency: Валюта для фільтрації (наприклад, 'USDT')
        :return: Список доступних символів
        """
        try:
            all_symbols = await self.exchange_client.fetch_available_symbols()

            # Фільтрація за вказаною валютою
            if quote_currency:
                filtered_symbols = [s for s in all_symbols if s.endswith(f'/{quote_currency}')]
                return filtered_symbols
            else:
                return all_symbols

        except Exception as e:
            print(f"Помилка при отриманні списку символів: {e}")
            return []
