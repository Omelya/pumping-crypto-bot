# !/usr/bin/env python
"""
Головний скрипт для запуску криптовалютного детектора підозрілої активності.
Підтримує як онлайн моніторинг, так і бектестинг на історичних даних.
"""

import asyncio
import argparse
import os
import logging

from datetime import datetime
from dotenv import load_dotenv
from crypto_detector.core.detector import CryptoActivityDetector
from crypto_detector.core.adaptive_detector import AdaptiveCryptoDetector
from crypto_detector.backtest.adaptive_backtest import AdaptiveBacktester

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Завантаження змінних середовища
load_dotenv()


def get_alert_threshold_for_symbol(symbol):
    """
    Визначення порогу тривоги для конкретного символу на основі категорії токена

    :param symbol: Символ криптовалюти
    :return: Поріг тривоги для цього символу
    """
    from crypto_detector.config.settings import TOKEN_THRESHOLDS, TOKEN_CATEGORIES

    base_currency = symbol.split('/')[0]

    # Визначення категорії токена
    category = 'other'  # За замовчуванням
    for cat, tokens in TOKEN_CATEGORIES.items():
        if base_currency in tokens:
            category = cat
            break

    # Повернення порогу з налаштувань
    return TOKEN_THRESHOLDS.get(category, TOKEN_THRESHOLDS['other'])

async def run_backtest(args):
    """
    Запуск бектестингу на історичних даних

    :param args: Параметри командного рядка
    """
    logger.info(f"Запуск бектестингу для {args.symbol} з {args.start_date} по {args.end_date}")

    # Ініціалізація детектора
    detector = CryptoActivityDetector(
        exchange_id=args.exchange,
        threshold_multiplier=args.threshold,
        lookback_period=args.lookback
    )

    # Ініціалізація адаптивного бектестера
    backtester = AdaptiveBacktester(detector, data_dir=args.data_dir)

    # Запуск адаптивного бектестингу
    if args.symbol == "all":
        symbols = [
            'SOL/USDT',
            'PEPE/USDT',
            'BTC/USDT',
            'ETH/USDT',
            'BNB/USDT',
            'AVAX/USDT',
            'DOGE / USDT',
            'CTT/USDT',
            '1000APUSDT',
            'A8/USDT',
            'ADA/USDT',
            'ALGO/USDT',
            # 'AUCTIONUSDT',
            # 'CAKE/USDT',
            'DOT/USDT',
            'GALA/USDT',
            'LAI/USDT',
            '1000XUSDT',
            'MAVIA/USDT',
            'MKR/USDT',
            'NOT/USDT',
            'SHIB/USDT',
            'SUI/USDT',
            'TON/USDT',
            'TRX/USDT',
            'XLM/USDT',
            'XRP/USDT',
            'YFI/USDT',
            'LINK/USDT',
            'BAND/USDT',
            'NMR/USDT',
            'FET/USDT',
            'CRO/USDT',
            'KCS/USDT',
        ]

        if args.symbols_file and os.path.exists(args.symbols_file):
            with open(args.symbols_file, 'r') as f:
                symbols = [line.strip() for line in f.readlines() if line.strip()]

        results = {}
        for sym in symbols:
            logger.info(f"Запуск бектестингу для {sym}...")
            try:
                sym_results = await backtester.backtest_with_adaptive_learning(
                    sym, args.start_date, args.end_date,
                    min_price_change=args.min_price_change,
                    window_hours=args.window_hours
                )
                if sym_results:
                    if args.visualize:
                        backtester.visualize_comparative_results(sym)
                    results[sym] = sym_results
            except Exception as e:
                logger.error(f"Помилка при бектестингу {sym}: {str(e)}")
                continue

        # Зберігаємо фінальні результати
        backtester.save_comparative_results(args.output)
        logger.info(f"Результати збережено у {args.output}")
    else:
        # Якщо потрібно тестувати один символ
        results = await backtester.backtest_with_adaptive_learning(
            args.symbol, args.start_date, args.end_date,
            min_price_change=args.min_price_change,
            window_hours=args.window_hours
        )

        if results and args.visualize:
            backtester.visualize_comparative_results(args.symbol)

        if results:
            backtester.save_comparative_results(args.output)
            logger.info(f"Результати збережено у {args.output}")

    # Навчання ML моделей, якщо вказано
    if args.train_ml:
        logger.info("Запуск навчання ML моделей на основі результатів бектестингу...")
        try:
            backtester.train_all_ml_models()
            logger.info("Навчання ML моделей завершено успішно.")
        except Exception as e:
            logger.error(f"Помилка при навчанні ML моделей: {str(e)}")

    return results


async def run_monitor(args):
    """
    Запуск онлайн-моніторингу криптовалютних ринків

    :param args: Параметри командного рядка
    """
    logger.info(f"Запуск онлайн-моніторингу для {args.symbol} на біржі {args.exchange}")

    # Ініціалізація детектора
    detector = CryptoActivityDetector(
        exchange_id=args.exchange,
        threshold_multiplier=args.threshold,
        lookback_period=args.lookback,
        alert_threshold=args.alert_threshold
    )

    # Якщо використовуємо адаптивний детектор
    if args.adaptive:
        detector = AdaptiveCryptoDetector(detector)
        logger.info("Використовується адаптивний детектор")

    # Отримання списку символів для моніторингу
    symbols_to_monitor = []

    if args.symbol == "all":
        # Отримання всіх доступних символів від біржі
        all_symbols = await detector.fetch_available_symbols()
        # Фільтрація для отримання тільки USDT пар
        symbols_to_monitor = [s for s in all_symbols if s.endswith('/USDT')]
    elif args.symbols_file and os.path.exists(args.symbols_file):
        # Завантаження списку символів з файлу
        with open(args.symbols_file, 'r') as f:
            symbols_to_monitor = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Використання одного вказаного символу
        symbols_to_monitor = [args.symbol]

    logger.info(f"Моніторинг {len(symbols_to_monitor)} символів: {', '.join(symbols_to_monitor[:5])}...")

    # Цикл моніторингу
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"Ітерація {iteration} моніторингу...")

        for symbol in symbols_to_monitor:
            try:
                # Отримати динамічний поріг для конкретної монети
                dynamic_threshold = get_alert_threshold_for_symbol(symbol)

                # Аналіз символу
                result = await detector.analyze_token(symbol, symbols_to_monitor)

                # Перевірка ймовірності з динамічним порогом
                if result['probability_score'] > dynamic_threshold:
                    logger.warning(f"ТРИВОГА! Виявлено підозрілу активність для {symbol}!")
                    logger.warning(f"Ймовірність: {result['probability_score']:.2f}, Поріг: {dynamic_threshold:.2f}")

                    # Запис детальної інформації
                    alert_file = os.path.join(args.output_dir,
                                              f"alert_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(alert_file, 'w') as f:
                        import json
                        json.dump(result, f, indent=4, default=str)

                elif result['probability_score'] > dynamic_threshold * 0.8:  # 80% від порогу
                    # Помірний рівень ризику
                    logger.info(
                        f"Увага: підвищена активність для {symbol}, ймовірність: {result['probability_score']:.2f}, поріг: {dynamic_threshold:.2f}")
                else:
                    # Нормальний рівень активності
                    logger.debug(
                        f"Нормальна активність для {symbol}, ймовірність: {result['probability_score']:.2f}, поріг: {dynamic_threshold:.2f}")

            except Exception as e:
                logger.error(f"Помилка при аналізі {symbol}: {str(e)}")
                continue

        # Пауза перед наступною ітерацією
        logger.info(f"Пауза {args.interval} секунд...")
        await asyncio.sleep(args.interval)

async def run_train(args):
    """
    Запуск режиму навчання ML моделей

    :param args: Параметри командного рядка
    """
    logger.info("Запуск режиму навчання ML моделей")

    # Ініціалізація детектора
    detector = CryptoActivityDetector(
        exchange_id='binance',
        threshold_multiplier=2.0,
        lookback_period=24
    )

    # Ініціалізація адаптивного детектора
    adaptive_detector = AdaptiveCryptoDetector(detector, model_dir=os.path.join(args.data_dir, "models"))

    # Ініціалізація бектестера для використання його методу train_all_ml_models
    backtester = AdaptiveBacktester(detector, data_dir=args.data_dir)
    backtester.adaptive_detector = adaptive_detector

    try:
        # Запуск навчання всіх моделей
        backtester.train_all_ml_models()
        logger.info("Навчання ML моделей завершено успішно.")
    except Exception as e:
        logger.error(f"Помилка при навчанні ML моделей: {str(e)}")

async def main():
    """
    Головна функція для парсингу аргументів та запуску відповідного режиму
    """
    parser = argparse.ArgumentParser(description='Криптовалютний детектор підозрілої активності')

    # Загальні параметри (без символу)
    parser.add_argument('--exchange', type=str, default='binance',
                        help='ID біржі для CCXT (binance, bybit, kucoin, тощо)')
    parser.add_argument('--threshold', type=float, default=2.0, help='Множник для визначення незвичайної активності')
    parser.add_argument('--lookback', type=int, default=24, help='Період аналізу історичних даних в годинах')
    parser.add_argument('--symbols-file', type=str, help='Файл зі списком символів для аналізу')

    # Створення підпарсерів для різних режимів
    subparsers = parser.add_subparsers(dest='mode', help='Режим роботи')

    # Налаштування для режиму бектестингу
    backtest_parser = subparsers.add_parser('backtest', help='Режим бектестингу на історичних даних')
    backtest_parser.add_argument('--symbol', type=str, default='BTC/USDT',
                                 help='Символ криптовалюти або "all" для всіх')
    backtest_parser.add_argument('--start-date', type=str, required=True, help='Початкова дата у форматі YYYY-MM-DD')
    backtest_parser.add_argument('--end-date', type=str, required=True, help='Кінцева дата у форматі YYYY-MM-DD')
    backtest_parser.add_argument('--min-price-change', type=float, default=5.0,
                                 help='Мінімальна зміна ціни для позначення події')
    backtest_parser.add_argument('--window-hours', type=int, default=24, help='Вікно для аналізу в годинах')
    backtest_parser.add_argument('--data-dir', type=str, default='historical_data',
                                 help='Директорія для історичних даних')
    backtest_parser.add_argument('--output', type=str, default='backtest_results.json',
                                 help='Файл для збереження результатів')
    backtest_parser.add_argument('--visualize', action='store_true', help='Візуалізувати результати')
    backtest_parser.add_argument('--train-ml', action='store_true', help='Тренувати ML моделі після бектестингу')

    # Налаштування для режиму моніторингу
    monitor_parser = subparsers.add_parser('monitor', help='Режим онлайн-моніторингу криптовалютних ринків')
    monitor_parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Символ криптовалюти або "all" для всіх')
    monitor_parser.add_argument('--interval', type=int, default=300, help='Інтервал між перевірками в секундах')
    monitor_parser.add_argument('--adaptive', action='store_true', help='Використовувати адаптивний детектор')
    monitor_parser.add_argument('--alert-threshold', type=float, default=0.35, help='Поріг для створення сповіщень')
    monitor_parser.add_argument('--output-dir', type=str, default='alerts', help='Директорія для збереження сповіщень')

    # Додамо новий режим для окремого тренування ML моделей
    train_parser = subparsers.add_parser('train', help='Режим навчання ML моделей')
    train_parser.add_argument('--data-dir', type=str, default='historical_data',
                              help='Директорія з даними для навчання')
    train_parser.add_argument('--symbols-file', type=str, help='Файл зі списком символів для навчання')
    train_parser.add_argument('--visualize', action='store_true', help='Візуалізувати результати навчання')
    train_parser.add_argument('--model-type', type=str, default='gradient_boosting',
                              choices=['gradient_boosting', 'random_forest'],
                              help='Тип ML моделі для навчання')

    args = parser.parse_args()

    # Створення директорій для виведення, якщо вони не існують
    if args.mode == 'monitor' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Запуск відповідного режиму
    if args.mode == 'backtest':
        await run_backtest(args)
    elif args.mode == 'monitor':
        await run_monitor(args)
    elif args.mode == 'train':
        await run_train(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
