import ccxt
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import asyncio
import time
from datetime import datetime, timedelta

from crypto_detector.utils.logging_utils import get_logger

# Завантаження змінних середовища
load_dotenv()


class ExchangeClient:
    """
    Клас для взаємодії з API криптовалютних бірж через бібліотеку CCXT.
    Відповідає за отримання даних з біржі та абстрагує деталі комунікації.
    """

    def __init__(self, exchange_id='binance', api_key=None, api_secret=None):
        """
        Ініціалізація клієнта біржі

        :param exchange_id: ID біржі для CCXT (binance, bybit, kucoin, okx, etc.)
        :param api_key: API ключ для біржі (якщо не вказано, береться з змінних середовища)
        :param api_secret: API секрет для біржі (якщо не вказано, береться з змінних середовища)
        """
        self.logger = get_logger(f"exchange_client_{exchange_id}")

        # Отримання API ключів з параметрів або змінних середовища
        self.api_key = api_key or os.getenv(f"{exchange_id.upper()}_API_KEY")
        self.api_secret = api_secret or os.getenv(f"{exchange_id.upper()}_API_SECRET")

        if not self.api_key or not self.api_secret:
            self.logger.warning(
                f"API ключі не знайдено. Доступ буде обмежений публічними API. "
                f"Додайте ключі як параметри або в .env файл як {exchange_id.upper()}_API_KEY та {exchange_id.upper()}_API_SECRET"
            )

        self.exchange_id = exchange_id

        # Ініціалізація біржового API через CCXT
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,  # Важливо для запобігання банам за API ліміти
            'timeout': 30000,  # Таймаут 30 секунд
            'options': {
                'defaultType': 'spot',  # За замовчуванням - спотовий ринок
            }
        })

        # Кеш для зменшення кількості запитів
        self.cache = {
            'symbols': None,
            'markets': None,
            'ohlcv': {}
        }

        # Час останнього оновлення кешу
        self.cache_updated = {
            'symbols': None,
            'markets': None
        }

        self.logger.info(f"Ініціалізовано клієнт для біржі {exchange_id}")

    async def fetch_ohlcv(self, symbol, timeframe='5m', since=None, limit=None):
        """
        Отримання OHLCV даних (Open, High, Low, Close, Volume) через CCXT

        :param symbol: Символ криптовалюти (наприклад, 'BTC/USDT')
        :param timeframe: Інтервал часу для даних
        :param since: Початкова часова мітка (timestamp) в мілісекундах
        :param limit: Кількість свічок для отримання
        :return: DataFrame з OHLCV даними
        """
        # Створення ключа для кешу
        cache_key = f"{symbol}_{timeframe}_{since}_{limit}"

        # Перевірка кешу
        if cache_key in self.cache['ohlcv']:
            self.logger.debug(f"Використання кешованих OHLCV даних для {symbol}")
            return self.cache['ohlcv'][cache_key]

        # Обмеження кількості свічок для запобігання занадто великим запитам
        max_limit = 1000
        if limit and limit > max_limit:
            self.logger.warning(f"Обмеження limit до {max_limit} для {symbol}")
            limit = max_limit

        try:
            # Використання асинхронного методу, якщо він підтримується
            if hasattr(self.exchange, 'fetch_ohlcv_async'):
                ohlcv = await self.exchange.fetch_ohlcv_async(symbol, timeframe, since, limit)
            else:
                # Якщо асинхронний метод не підтримується, виконуємо синхронний у окремому потоці
                loop = asyncio.get_event_loop()
                ohlcv = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                )

            # Перетворення в DataFrame
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

                # Кешування результату
                self.cache['ohlcv'][cache_key] = df

                self.logger.debug(f"Отримано {len(df)} OHLCV свічок для {symbol}")
                return df
            else:
                self.logger.warning(f"Отримано порожній результат OHLCV для {symbol}")
                return pd.DataFrame()

        except ccxt.NetworkError as e:
            self.logger.error(f"Мережева помилка при отриманні OHLCV для {symbol}: {str(e)}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            self.logger.error(f"Помилка біржі при отриманні OHLCV для {symbol}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Невідома помилка при отриманні OHLCV для {symbol}: {str(e)}")
            return pd.DataFrame()

    async def fetch_trades(self, symbol, limit=100, since=None):
        """
        Отримання останніх угод для аналізу активності

        :param symbol: Символ криптовалюти
        :param limit: Кількість останніх угод
        :param since: Початкова часова мітка (timestamp) в мілісекундах
        :return: DataFrame з угодами
        """
        try:
            # Використання асинхронного методу, якщо він підтримується
            if hasattr(self.exchange, 'fetch_trades_async'):
                trades = await self.exchange.fetch_trades_async(symbol, limit=limit, since=since)
            else:
                # Якщо асинхронний метод не підтримується, виконуємо синхронний у окремому потоці
                loop = asyncio.get_event_loop()
                trades = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_trades(symbol, limit=limit, since=since)
                )

            # Перетворення в DataFrame
            if trades and len(trades) > 0:
                df = pd.DataFrame(trades)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)

                self.logger.debug(f"Отримано {len(df)} угод для {symbol}")
                return df
            else:
                self.logger.warning(f"Отримано порожній результат угод для {symbol}")
                return pd.DataFrame()
        except ccxt.NetworkError as e:
            self.logger.error(f"Мережева помилка при отриманні угод для {symbol}: {str(e)}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            self.logger.error(f"Помилка біржі при отриманні угод для {symbol}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Невідома помилка при отриманні угод для {symbol}: {str(e)}")
            return pd.DataFrame()

    async def fetch_order_book(self, symbol, limit=20):
        """
        Отримання книги ордерів

        :param symbol: Символ криптовалюти
        :param limit: Глибина книги ордерів
        :return: Дані книги ордерів
        """
        try:
            # Використання асинхронного методу, якщо він підтримується
            if hasattr(self.exchange, 'fetch_order_book_async'):
                order_book = await self.exchange.fetch_order_book_async(symbol, limit)
            else:
                # Якщо асинхронний метод не підтримується, виконуємо синхронний у окремому потоці
                loop = asyncio.get_event_loop()
                order_book = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_order_book(symbol, limit)
                )

            # Додавання часової мітки, якщо вона відсутня
            if 'timestamp' not in order_book or not order_book['timestamp']:
                order_book['timestamp'] = int(time.time() * 1000)

            self.logger.debug(f"Отримано книгу ордерів для {symbol} (глибина: {limit})")
            return order_book
        except ccxt.NetworkError as e:
            self.logger.error(f"Мережева помилка при отриманні книги ордерів для {symbol}: {str(e)}")
            return {'bids': [], 'asks': [], 'timestamp': int(time.time() * 1000)}
        except ccxt.ExchangeError as e:
            self.logger.error(f"Помилка біржі при отриманні книги ордерів для {symbol}: {str(e)}")
            return {'bids': [], 'asks': [], 'timestamp': int(time.time() * 1000)}
        except Exception as e:
            self.logger.error(f"Невідома помилка при отриманні книги ордерів для {symbol}: {str(e)}")
            return {'bids': [], 'asks': [], 'timestamp': int(time.time() * 1000)}

    async def fetch_available_symbols(self, quote_currency=None, cached=True):
        """
        Отримання списку доступних символів на біржі

        :param quote_currency: Фільтрація за котирувальною валютою (наприклад, 'USDT')
        :param cached: Використовувати кешовані дані, якщо доступні
        :return: Список доступних символів
        """
        # Перевірка кешу
        cache_expired = (
                self.cache['symbols'] is None or
                self.cache_updated['symbols'] is None or
                (datetime.now() - self.cache_updated['symbols']).total_seconds() > 3600  # 1 година
        )

        if not cached or cache_expired:
            try:
                # Отримання всіх ринків
                if hasattr(self.exchange, 'fetch_markets_async'):
                    markets = await self.exchange.fetch_markets_async()
                else:
                    loop = asyncio.get_event_loop()
                    markets = await loop.run_in_executor(None, lambda: self.exchange.fetch_markets())

                # Фільтрація активних ринків і отримання символів
                active_markets = [market for market in markets if market.get('active', True)]
                all_symbols = [market['symbol'] for market in active_markets]

                # Оновлення кешу
                self.cache['symbols'] = all_symbols
                self.cache['markets'] = active_markets
                self.cache_updated['symbols'] = datetime.now()
                self.cache_updated['markets'] = datetime.now()

                self.logger.info(f"Оновлено список символів: знайдено {len(all_symbols)} символів")
            except Exception as e:
                self.logger.error(f"Помилка при отриманні списку символів: {str(e)}")
                # Якщо є кешовані дані, використовуємо їх
                if self.cache['symbols'] is not None:
                    all_symbols = self.cache['symbols']
                    self.logger.warning("Використовуємо кешовані дані символів")
                else:
                    return []
        else:
            all_symbols = self.cache['symbols']
            self.logger.debug("Використання кешованих символів")

        # Фільтрація за котирувальною валютою
        if quote_currency:
            filtered_symbols = [s for s in all_symbols if s.endswith(f'/{quote_currency}')]
            self.logger.debug(f"Відфільтровано {len(filtered_symbols)} символів з {quote_currency}")
            return filtered_symbols
        else:
            return all_symbols

    async def fetch_ticker(self, symbol):
        """
        Отримання тікера для символу

        :param symbol: Символ криптовалюти
        :return: Дані тікера
        """
        try:
            # Використання асинхронного методу, якщо він підтримується
            if hasattr(self.exchange, 'fetch_ticker_async'):
                ticker = await self.exchange.fetch_ticker_async(symbol)
            else:
                # Якщо асинхронний метод не підтримується, виконуємо синхронний у окремому потоці
                loop = asyncio.get_event_loop()
                ticker = await loop.run_in_executor(None, lambda: self.exchange.fetch_ticker(symbol))

            self.logger.debug(f"Отримано тікер для {symbol}")
            return ticker
        except Exception as e:
            self.logger.error(f"Помилка при отриманні тікера для {symbol}: {str(e)}")
            return {}

    async def fetch_multi_ohlcv(self, symbols, timeframe='5m', since=None, limit=None):
        """
        Отримання OHLCV даних для кількох символів

        :param symbols: Список символів криптовалют
        :param timeframe: Інтервал часу для даних
        :param since: Початкова часова мітка (timestamp) в мілісекундах
        :param limit: Кількість свічок для отримання
        :return: Словник {символ: DataFrame з OHLCV даними}
        """
        # Обмеження кількості одночасних запитів
        max_concurrent = 5
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(symbol):
            async with semaphore:
                return symbol, await self.fetch_ohlcv(symbol, timeframe, since, limit)

        # Створення завдань для всіх символів
        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]

        # Виконання всіх завдань
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обробка результатів
        ohlcv_data = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Помилка при отриманні OHLCV даних: {str(result)}")
                continue

            symbol, df = result
            if not df.empty:
                ohlcv_data[symbol] = df

        self.logger.info(f"Отримано OHLCV дані для {len(ohlcv_data)} з {len(symbols)} символів")
        return ohlcv_data

    async def download_historical_data(self, symbol, start_date, end_date, timeframe='5m'):
        """
        Завантаження історичних даних за вказаний період

        :param symbol: Символ криптовалюти
        :param start_date: Початкова дата (рядок 'YYYY-MM-DD' або об'єкт datetime)
        :param end_date: Кінцева дата (рядок 'YYYY-MM-DD' або об'єкт datetime)
        :param timeframe: Інтервал часу для даних
        :return: DataFrame з історичними даними
        """
        # Конвертація дат в об'єкти datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Переконуємося, що кінцева дата не в майбутньому
        current_date = datetime.now()
        if end_date > current_date:
            end_date = current_date
            self.logger.warning(f"Кінцева дата змінена на поточну: {end_date.strftime('%Y-%m-%d')}")

        # Різниця в днях для оцінки обсягу даних
        days_diff = (end_date - start_date).days + 1

        # Оцінка кількості свічок
        tf_minutes = self.convert_timeframe_to_minutes(timeframe)
        estimated_candles = int(days_diff * 24 * 60 / tf_minutes)

        self.logger.info(
            f"Завантаження історичних даних для {symbol} з {start_date.strftime('%Y-%m-%d')} "
            f"по {end_date.strftime('%Y-%m-%d')} (приблизно {estimated_candles} свічок)"
        )

        # Обмеження для одного запиту
        max_candles_per_request = 1000

        # Якщо оцінка кількості свічок перевищує обмеження, розбиваємо на частини
        if estimated_candles > max_candles_per_request:
            # Розрахунок інтервалу для кожного запиту в днях
            interval_days = max(1, int(max_candles_per_request * tf_minutes / (24 * 60)))

            # Розбиття на частини
            current_start = start_date
            all_data = []

            while current_start < end_date:
                # Розрахунок кінця поточного інтервалу
                current_end = min(current_start + timedelta(days=interval_days), end_date)

                # Конвертація timestamp в мілісекунди для CCXT
                since = int(current_start.timestamp() * 1000)

                # Завантаження даних для поточного інтервалу
                self.logger.debug(
                    f"Завантаження даних для {symbol} з {current_start.strftime('%Y-%m-%d')} "
                    f"по {current_end.strftime('%Y-%m-%d')}"
                )

                interval_data = await self.fetch_ohlcv(symbol, timeframe, since=since, limit=max_candles_per_request)

                if not interval_data.empty:
                    all_data.append(interval_data)
                    self.logger.debug(f"Отримано {len(interval_data)} свічок")

                # Затримка для запобігання перевищенню лімітів API
                await asyncio.sleep(1)

                # Оновлення початкової дати для наступного інтервалу
                # Використовуємо останню отриману часову мітку + 1 свічка
                if not interval_data.empty:
                    last_timestamp = interval_data.index[-1]
                    # Додаємо інтервал часу для уникнення дублікатів
                    current_start = last_timestamp.to_pydatetime() + timedelta(minutes=tf_minutes)
                else:
                    # Якщо дані не отримано, рухаємося вперед на інтервал
                    current_start = current_end

            # Об'єднання всіх частин
            if all_data:
                combined_data = pd.concat(all_data)
                # Видалення дублікатів (може виникнути при розбитті на частини)
                combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
                combined_data.sort_index(inplace=True)

                # Фільтрація за вказаним діапазоном дат
                start_timestamp = pd.Timestamp(start_date)
                end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                filtered_data = combined_data[
                    (combined_data.index >= start_timestamp) &
                    (combined_data.index <= end_timestamp)
                    ]

                self.logger.info(f"Завантажено {len(filtered_data)} свічок для {symbol}")
                return filtered_data
            else:
                self.logger.warning(f"Не вдалося отримати дані для {symbol}")
                return pd.DataFrame()
        else:
            # Якщо кількість свічок в межах обмеження, робимо один запит
            since = int(start_date.timestamp() * 1000)
            data = await self.fetch_ohlcv(symbol, timeframe, since=since, limit=estimated_candles)

            if not data.empty:
                # Фільтрація за вказаним діапазоном дат
                start_timestamp = pd.Timestamp(start_date)
                end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                filtered_data = data[
                    (data.index >= start_timestamp) &
                    (data.index <= end_timestamp)
                    ]

                self.logger.info(f"Завантажено {len(filtered_data)} свічок для {symbol}")
                return filtered_data
            else:
                self.logger.warning(f"Не вдалося отримати дані для {symbol}")
                return pd.DataFrame()

    def convert_timeframe_to_minutes(self, timeframe):
        """
        Конвертація рядка timeframe у хвилини

        :param timeframe: Рядок з таймфреймом (e.g., '5m', '1h', '1d')
        :return: Кількість хвилин
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 24 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60
        else:
            raise ValueError(f"Непідтримуваний timeframe: {timeframe}")

    def clear_cache(self, cache_type=None):
        """
        Очищення кешу

        :param cache_type: Тип кешу для очищення ('symbols', 'markets', 'ohlcv', або None для всіх)
        """
        if cache_type is None or cache_type == 'symbols':
            self.cache['symbols'] = None
            self.cache_updated['symbols'] = None
            self.logger.debug("Очищено кеш символів")

        if cache_type is None or cache_type == 'markets':
            self.cache['markets'] = None
            self.cache_updated['markets'] = None
            self.logger.debug("Очищено кеш ринків")

        if cache_type is None or cache_type == 'ohlcv':
            self.cache['ohlcv'] = {}
            self.logger.debug("Очищено кеш OHLCV даних")
