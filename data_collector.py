#!/usr/bin/env python3
"""
Скрипт для збору даних про криптовалюти та їх збереження для аналізу кореляцій.
Працює як окремий процес, який збирає та зберігає дані в спільну базу даних.

Використання:
    python data_collector.py --exchange binance --db crypto_data.db --symbols-limit 100 --interval 60
"""
import asyncio
import logging
import argparse
import time
import sys
import os
import signal
import random

# Додаємо шлях до батьківської директорії для імпорту модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Імпортуємо модулі
from crypto_detector.analysis.correlation_analyzer import CorrelationAnalyzer
from crypto_detector.data.exchange_client import ExchangeClient

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataCollector")

# Глобальні змінні для зупинки програми
should_exit = False

# Визначаємо ліміти API для різних бірж
API_RATE_LIMITS = {
    'bybit': {
        'requests_per_minute': 60,
        'batch_size': 5,  # Зменшено розмір пакету
        'retry_delay': 10,  # Секунд до повторної спроби
        'concurrent_requests': 3  # Зменшено кількість одночасних запитів
    },
    'binance': {
        'requests_per_minute': 1200,
        'batch_size': 20,
        'retry_delay': 1,
        'concurrent_requests': 10
    },
    'kucoin': {
        'requests_per_minute': 180,
        'batch_size': 10,
        'retry_delay': 5,
        'concurrent_requests': 5
    },
    'okx': {
        'requests_per_minute': 600,
        'batch_size': 15,
        'retry_delay': 2,
        'concurrent_requests': 8
    },
    # Значення за замовчуванням для інших бірж
    'default': {
        'requests_per_minute': 60,
        'batch_size': 5,
        'retry_delay': 10,
        'concurrent_requests': 3
    }
}


def signal_handler(sig, frame):
    """Обробник сигналу для коректного завершення"""
    global should_exit
    logger.info("Отримано сигнал завершення. Зупиняю роботу...")
    should_exit = True


# Реєструємо обробники сигналів
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class RateLimiter:
    """Клас для керування лімітами запитів API"""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
        self.semaphore = asyncio.Semaphore(requests_per_minute)

    async def acquire(self):
        """Отримання дозволу на виконання запиту з урахуванням обмежень"""
        await self.semaphore.acquire()

        # Очищення старих часових міток
        now = time.time()
        minute_ago = now - 60
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]

        # Якщо досягнуто ліміт запитів за хвилину, чекаємо
        if len(self.request_timestamps) >= self.requests_per_minute:
            oldest = self.request_timestamps[0]
            wait_time = max(0, 60 - (now - oldest))
            if wait_time > 0:
                logger.debug(f"Досягнуто ліміт API, очікування {wait_time:.2f} секунд")
                await asyncio.sleep(wait_time)

        # Додаємо поточну часову мітку
        self.request_timestamps.append(time.time())

    def release(self):
        """Звільнення семафора після виконання запиту"""
        self.semaphore.release()


async def fetch_with_rate_limit(rate_limiter: RateLimiter, func, *args, **kwargs):
    """
    Виконує функцію з урахуванням обмежень API

    :param rate_limiter: Екземпляр RateLimiter
    :param func: Функція для виконання
    :param args: Позиційні аргументи для функції
    :param kwargs: Іменовані аргументи для функції
    :return: Результат виконання функції або None у разі помилки
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            await rate_limiter.acquire()
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            retry_count += 1

            # Перевіряємо, чи помилка пов'язана з обмеженням API
            rate_limit_errors = ["rate limit", "too many", "too frequent", "exceeded", "limit exceeded"]
            is_rate_limit_error = any(err in str(e).lower() for err in rate_limit_errors)

            if is_rate_limit_error:
                # Експоненціальна затримка при обмеженні API
                wait_time = 2 ** retry_count + random.uniform(0, 1)
                logger.warning(
                    f"Обмеження API при виклику {func.__name__}, очікування {wait_time:.2f} секунд (спроба {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            elif retry_count < max_retries:
                # Інші помилки з меншою затримкою
                logger.error(f"Помилка при виклику {func.__name__}: {e}, повторна спроба {retry_count}/{max_retries}")
                await asyncio.sleep(1)
            else:
                logger.error(f"Не вдалося виконати {func.__name__} після {max_retries} спроб: {e}")
        finally:
            rate_limiter.release()

    return None


async def collect_initial_data(exchange_client, analyzer, symbols, rate_limiter, exchange_limits, timeframe='5m',
                               limit=100):
    """
    Початкове наповнення бази даних історичними даними

    :param exchange_client: Екземпляр ExchangeClient
    :param analyzer: Екземпляр CorrelationAnalyzer
    :param symbols: Список символів для збору даних
    :param rate_limiter: Екземпляр RateLimiter
    :param exchange_limits: Обмеження API для біржі
    :param timeframe: Часовий інтервал
    :param limit: Кількість свічок для кожного символу
    """
    logger.info(f"Початкове наповнення бази даними для {len(symbols)} символів")

    # Зменшуємо розмір початкового завантаження для уникнення перевантаження API
    initial_limit = min(limit, 200)

    # Використовуємо пакетний підхід для уникнення перевантаження API
    batch_size = exchange_limits['batch_size']

    # Випадковий порядок символів для рівномірного розподілу важливих монет
    random.shuffle(symbols)

    for i in range(0, len(symbols), batch_size):
        if should_exit:
            logger.info("Отримано сигнал завершення. Зупиняю початкове наповнення...")
            break

        batch = symbols[i:i + batch_size]
        logger.info(
            f"Обробляю пакет {i // batch_size + 1}/{(len(symbols) - 1) // batch_size + 1} ({len(batch)} символів)")

        # Завантажуємо дані для кожного символу послідовно для зменшення навантаження
        for symbol in batch:
            if should_exit:
                break

            # Використовуємо RateLimiter для контролю частоти запитів
            ohlcv_data = await fetch_with_rate_limit(
                rate_limiter,
                exchange_client.fetch_ohlcv,
                symbol, timeframe, limit=initial_limit
            )

            if ohlcv_data is None or ohlcv_data.empty:
                logger.warning(f"Не отримано даних для {symbol}")
                continue

            # Додаємо дані в аналізатор
            for timestamp, row in ohlcv_data.iterrows():
                analyzer.add_market_data(
                    symbol,
                    row['close'],
                    row['volume'],
                    timestamp.to_pydatetime()
                )

            logger.debug(f"Додано {len(ohlcv_data)} свічок для {symbol}")

            # Невелика затримка між запитами для одного символу
            await asyncio.sleep(0.2)

        # Затримка між пакетами для зменшення навантаження на API
        wait_time = random.uniform(1, 3)
        logger.debug(f"Затримка між пакетами: {wait_time:.2f} секунд")
        await asyncio.sleep(wait_time)

    # Виводимо статистику
    stats = analyzer.get_database_stats()
    logger.info(f"Початкове наповнення завершено. Статистика бази даних:")
    logger.info(f"Записів: {stats['total_records']}")
    logger.info(f"Унікальних символів: {stats['unique_symbols']}")
    logger.info(f"Розмір бази даних: {stats['database_size_mb']:.2f} MB")


async def process_batch_for_continuous_collection(exchange_client, analyzer, batch, rate_limiter):
    """
    Обробка пакету символів для безперервного збору даних

    :param exchange_client: Екземпляр ExchangeClient
    :param analyzer: Екземпляр CorrelationAnalyzer
    :param batch: Список символів для обробки
    :param rate_limiter: Екземпляр RateLimiter
    :return: Кількість успішно оброблених символів
    """
    successful_updates = 0

    for symbol in batch:
        if should_exit:
            break

        # Отримуємо тікер з урахуванням лімітів API
        ticker = await fetch_with_rate_limit(
            rate_limiter,
            exchange_client.fetch_ticker,
            symbol
        )

        if not ticker:
            continue

        # Додаємо дані в аналізатор
        price = ticker.get('last', None)
        volume = ticker.get('quoteVolume', ticker.get('volume', 0))

        if price:
            analyzer.add_market_data(symbol, price, volume)
            successful_updates += 1

        # Невелика випадкова затримка між запитами
        await asyncio.sleep(random.uniform(0.1, 0.3))

    return successful_updates


async def continuous_data_collection(exchange_client, analyzer, symbols, rate_limiter, exchange_limits, interval=60):
    """
    Безперервний збір даних тікерів з урахуванням обмежень API

    :param exchange_client: Екземпляр ExchangeClient
    :param analyzer: Екземпляр CorrelationAnalyzer
    :param symbols: Список символів для збору даних
    :param rate_limiter: Екземпляр RateLimiter
    :param exchange_limits: Обмеження API для біржі
    :param interval: Інтервал між збором даних (секунди)
    """
    logger.info(f"Початок безперервного збору даних для {len(symbols)} символів з інтервалом {interval} секунд")

    # Пріоритетні символи завжди оновлюються першими
    priority_coins = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
                      "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT", "DOT/USDT"]

    # Максимальна кількість символів для оновлення в одному циклі
    max_symbols_per_cycle = min(len(symbols), exchange_limits['requests_per_minute'] // 2)

    # Група символів для OHLCV оновлення (невелика підмножина)
    ohlcv_group_size = max(1, max_symbols_per_cycle // 10)

    # Розмір пакету для обробки
    batch_size = exchange_limits['batch_size']

    while not should_exit:
        start_time = time.time()
        cycle_successful_updates = 0

        try:
            # Формуємо список символів для поточного циклу
            # Спочатку пріоритетні символи, потім випадкова підвибірка інших
            current_cycle_symbols = []

            # Додаємо пріоритетні символи
            for coin in priority_coins:
                if coin in symbols:
                    current_cycle_symbols.append(coin)

            # Додаємо випадкову підвибірку решти символів
            remaining_symbols = [s for s in symbols if s not in current_cycle_symbols]
            random.shuffle(remaining_symbols)

            remaining_slots = max_symbols_per_cycle - len(current_cycle_symbols)
            if remaining_slots > 0:
                current_cycle_symbols.extend(remaining_symbols[:remaining_slots])

            logger.debug(f"Оновлення даних для {len(current_cycle_symbols)} символів у поточному циклі")

            # Розбиваємо на пакети
            for i in range(0, len(current_cycle_symbols), batch_size):
                if should_exit:
                    break

                batch = current_cycle_symbols[i:i + batch_size]

                # Обробляємо пакет
                batch_updates = await process_batch_for_continuous_collection(
                    exchange_client, analyzer, batch, rate_limiter
                )

                cycle_successful_updates += batch_updates

                # Затримка між пакетами
                await asyncio.sleep(random.uniform(0.5, 1.5))

            # OHLCV оновлення для підмножини символів
            if not should_exit and cycle_successful_updates > 0:
                # Вибираємо символи для OHLCV оновлення
                ohlcv_symbols = priority_coins[:3]  # Завжди оновлюємо BTC, ETH, BNB

                # Додаємо випадкову підвибірку з інших символів
                random_ohlcv = random.sample(
                    [s for s in current_cycle_symbols if s not in ohlcv_symbols],
                    min(ohlcv_group_size - len(ohlcv_symbols), len(current_cycle_symbols) - len(ohlcv_symbols))
                )
                ohlcv_symbols.extend(random_ohlcv)

                logger.debug(f"Оновлення OHLCV даних для {len(ohlcv_symbols)} символів")

                # Оновлюємо OHLCV дані
                for symbol in ohlcv_symbols:
                    if should_exit:
                        break

                    ohlcv_data = await fetch_with_rate_limit(
                        rate_limiter,
                        exchange_client.fetch_ohlcv,
                        symbol, '5m', limit=5
                    )

                    if ohlcv_data is not None and not ohlcv_data.empty:
                        # Додаємо останню свічку в аналізатор
                        last_row = ohlcv_data.iloc[-1]
                        analyzer.add_market_data(
                            symbol,
                            last_row['close'],
                            last_row['volume'],
                            ohlcv_data.index[-1].to_pydatetime()
                        )

                    # Затримка між запитами
                    await asyncio.sleep(random.uniform(0.3, 0.7))

            logger.info(f"Цикл завершено: оновлено {cycle_successful_updates} символів")

        except Exception as e:
            logger.error(f"Помилка під час циклу збору даних: {e}")

        # Очікуємо до наступного інтервалу
        elapsed = time.time() - start_time
        wait_time = max(0, interval - elapsed)

        if wait_time > 0 and not should_exit:
            logger.info(f"Очікую {wait_time:.2f} секунд до наступного циклу збору даних")
            await asyncio.sleep(wait_time)


async def fetch_symbols_safely(exchange_client, rate_limiter, quote_currency='USDT'):
    """
    Безпечне отримання списку символів з урахуванням лімітів API

    :param exchange_client: Екземпляр ExchangeClient
    :param rate_limiter: Екземпляр RateLimiter
    :param quote_currency: Котирувальна валюта для фільтрації
    :return: Список доступних символів
    """
    logger.info(f"Отримання списку символів для {quote_currency}...")

    # Отримуємо список символів з урахуванням лімітів API
    available_symbols = await fetch_with_rate_limit(
        rate_limiter,
        exchange_client.fetch_available_symbols,
        quote_currency=quote_currency
    )

    if not available_symbols:
        logger.error("Не вдалося отримати список символів")
        return []

    # Фільтруємо тільки активні ринки з USDT
    symbols = [
        symbol for symbol in available_symbols
        if f'/{quote_currency}' in symbol and
           not symbol.startswith('BEAR/') and
           not symbol.startswith('BULL/')
    ]

    logger.info(f"Отримано {len(symbols)} символів для аналізу")
    return symbols


async def main():
    """
    Головна функція програми
    """
    parser = argparse.ArgumentParser(description='Збір даних про криптовалюти для аналізу кореляцій')
    parser.add_argument('--exchange', type=str, default='binance', help='Біржа для отримання даних')
    parser.add_argument('--symbols-limit', type=int, default=100, help='Кількість символів для аналізу')
    parser.add_argument('--interval', type=int, default=60, help='Інтервал між запитами (секунди)')
    parser.add_argument('--skip-initial', action='store_true', help='Пропустити початкове наповнення бази')
    parser.add_argument('--market-type', type=str, default='spot', choices=['spot', 'future'],
                        help='Тип ринку (spot або future)')
    parser.add_argument('--quote-currency', type=str, default='USDT', help='Котирувальна валюта (USDT, USDC, etc.)')

    args = parser.parse_args()

    # Отримання обмежень API для вибраної біржі
    exchange_limits = API_RATE_LIMITS.get(args.exchange.lower(), API_RATE_LIMITS['default'])

    # Створюємо RateLimiter для контролю частоти запитів
    rate_limiter = RateLimiter(exchange_limits['requests_per_minute'])

    # Ініціалізація аналізатора кореляцій
    analyzer = CorrelationAnalyzer()

    # Створення екземпляра ExchangeClient
    exchange_client = ExchangeClient(args.exchange)

    # Встановлюємо тип ринку
    if hasattr(exchange_client.exchange, 'options'):
        exchange_client.exchange.options['defaultType'] = args.market_type

    try:
        # Безпечне отримання списку доступних символів
        symbols = await fetch_symbols_safely(exchange_client, rate_limiter, args.quote_currency)

        if not symbols:
            logger.error("Не отримано символів для аналізу. Завершення програми.")
            return

        # Обмежуємо кількість символів
        symbols = symbols[:args.symbols_limit]
        logger.info(f"Вибрано {len(symbols)} символів для аналізу")

        # Початкове наповнення бази даних
        if not args.skip_initial:
            await collect_initial_data(
                exchange_client, analyzer, symbols, rate_limiter, exchange_limits, '5m', 100
            )
        else:
            logger.info("Пропускаю початкове наповнення бази даних")

        # Запускаємо безперервний збір даних
        await continuous_data_collection(
            exchange_client, analyzer, symbols, rate_limiter, exchange_limits, args.interval
        )

    except Exception as e:
        logger.error(f"Помилка виконання: {e}")
        raise
    finally:
        logger.info("Завершення роботи")


if __name__ == "__main__":
    # Запускаємо головну функцію
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Програму зупинено користувачем")
    except Exception as e:
        logger.error(f"Критична помилка: {e}")
    finally:
        logger.info("Програма завершена")
