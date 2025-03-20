import asyncio
from datetime import datetime

from crypto_detector.data.exchange_client import ExchangeClient
from crypto_detector.analysis.volume_analyzer import VolumeAnalyzer
from crypto_detector.analysis.price_analyzer import PriceAnalyzer
from crypto_detector.analysis.orderbook_analyzer import OrderBookAnalyzer
from crypto_detector.analysis.time_pattern_analyzer import TimePatternAnalyzer
from crypto_detector.analysis.correlation_analyzer import CorrelationAnalyzer
from crypto_detector.managers.social_media_manager import SocialMediaManager


class CryptoActivityDetector:
    """
    Основний клас для виявлення підозрілої активності на криптовалютних ринках.
    Включає виявлення pump-and-dump схем та інших аномалій.
    """

    def __init__(self, exchange_id='binance', threshold_multiplier=2, lookback_period=24,
                 api_key=None, api_secret=None, alert_threshold=0.35):
        """
        Ініціалізація детектора підвищеної активності криптовалют

        :param exchange_id: ID біржі для CCXT (binance, kucoin, okx, etc.)
        :param threshold_multiplier: Множник для визначення незвичайної активності
        :param lookback_period: Кількість годин для аналізу історичних даних
        :param api_key: API ключ для біржі
        :param api_secret: API секрет для біржі
        :param alert_threshold: Поріг для створення сповіщень
        """
        # Ініціалізація клієнта біржі
        self.exchange_client = ExchangeClient(exchange_id, api_key, api_secret)
        self.exchange = self.exchange_client.exchange  # Прямий доступ до CCXT для зворотної сумісності

        # Основні параметри
        self.exchange_id = exchange_id
        self.threshold_multiplier = threshold_multiplier
        self.lookback_period = lookback_period
        self.alert_threshold = alert_threshold

        # Ініціалізація аналізаторів
        self.volume_analyzer = VolumeAnalyzer(threshold_multiplier)
        self.price_analyzer = PriceAnalyzer()
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.time_pattern_analyzer = TimePatternAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()

        self.social_media_manager = SocialMediaManager()

        # Відстеження результатів
        self.alerts = []
        self.history = {}
        self.social_history = {}
        self.coin_history = {}

    async def fetch_ohlcv(self, symbol, timeframe='5m', limit=None):
        """
        Отримання OHLCV даних через клієнт біржі

        :param symbol: Символ криптовалюти
        :param timeframe: Інтервал часу для даних
        :param limit: Кількість свічок
        :return: DataFrame з даними
        """
        if limit is None:
            # Розрахунок кількості свічок на основі lookback_period
            minutes_in_timeframe = self.exchange_client.convert_timeframe_to_minutes(timeframe)
            limit = int((self.lookback_period * 60) / minutes_in_timeframe) + 10  # +10 для запасу

        return await self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def analyze_token(self, symbol, all_symbols=None):
        """
        Комплексний аналіз криптовалюти з покращеним виявленням pump-and-dump схем

        :param symbol: Символ криптовалюти з форматом CCXT (наприклад, 'BTC/USDT')
        :param all_symbols: Список всіх символів для аналізу корелляції (опціонально)
        :return: Dict з оцінкою ймовірності підвищення активності
        """
        if all_symbols is None:
            all_symbols = [symbol]

        # Отримання та аналіз даних
        tasks = [
            self.fetch_ohlcv(symbol, '5m'),
            self.exchange_client.fetch_order_book(symbol),
            self.social_media_manager.detect_social_media_mentions(symbol),
            self.correlation_analyzer.analyze_market_correlation(symbol, all_symbols),
            self.time_pattern_analyzer.check_time_pattern()
        ]

        # Асинхронно отримуємо всі дані
        ohlcv_data, order_book, social_data, correlation_data, time_pattern_data = await asyncio.gather(*tasks)

        # Аналіз отриманих даних
        volume_analysis = await self.volume_analyzer.detect_unusual_volume(ohlcv_data)
        price_analysis = await self.price_analyzer.detect_price_action(ohlcv_data)
        order_book_analysis = await self.orderbook_analyzer.analyze_order_book(order_book)

        # Розрахунок ймовірності підвищення активності
        signals = []
        confidence = 0.0

        # Сигнал 1: Аномальний обсяг торгів
        if volume_analysis['unusual_volume']:
            signals.append({
                'name': 'Аномальний обсяг торгів',
                'description': f"Поточний обсяг перевищує середній на {volume_analysis['percent_change']:.2f}%",
                'weight': 0.35
            })
            confidence += 0.35 * min(volume_analysis['percent_change'] / 80, 1.0) if volume_analysis[
                                                                                         'percent_change'] > 0 else 0

        # Сигнал 2: Цінова динаміка
        if price_analysis['price_action_signal']:
            signals.append({
                'name': 'Активна цінова динаміка',
                'description': f"Зміна ціни за 1 годину: {price_analysis['price_change_1h']:.2f}%, послідовних зростань: {price_analysis['consecutive_up']}",
                'weight': 0.25
            })
            confidence += 0.25 * min(abs(price_analysis['price_change_1h']) / 8, 1.0)

        # Сигнал 3: Дисбаланс книги ордерів
        if order_book_analysis['order_book_signal']:
            signals.append({
                'name': 'Дисбаланс книги ордерів',
                'description': f"Співвідношення ордерів купівлі/продажу: {order_book_analysis['buy_sell_ratio']:.2f}",
                'weight': 0.2
            })
            ratio = order_book_analysis['buy_sell_ratio']
            confidence += 0.2 * (min(ratio / 1.7, 1.0) if ratio > 1 else min(1 / ratio / 1.7, 1.0))

        # Сигнал 4: Підвищена соціальна активність
        if social_data['social_signal']:
            signals.append({
                'name': 'Підвищена активність у соціальних мережах',
                'description': f"Збільшення згадок на {social_data['percent_change']:.2f}% порівняно з середнім",
                'weight': 0.2
            })
            confidence += 0.2 * min(social_data['percent_change'] / 80, 1.0)

        # Сигнал 5: Часовий патерн
        if time_pattern_data['time_pattern_signal']:
            signals.append({
                'name': 'Підозрілий часовий патерн',
                'description': f"Поточний час відповідає високоризиковому періоду для pump-and-dump схем",
                'weight': 0.15
            })
            confidence += 0.15 * time_pattern_data['time_risk_score']

        # Сигнал 6: Кореляція з іншими ринками
        if correlation_data['correlation_signal']:
            correlated_coins = ', '.join(correlation_data['correlated_coins']) if correlation_data[
                'correlated_coins'] else "немає даних"
            signals.append({
                'name': 'Корельована активність з іншими монетами',
                'description': f"Виявлено схожу активність на інших монетах: {correlated_coins}",
                'weight': 0.15
            })
            confidence += 0.15

        # Сигнал 7: Прискорення об'єму
        if volume_analysis.get('volume_acceleration', 0) > 0.1:
            signals.append({
                'name': 'Прискорення зростання об\'єму',
                'description': f"Швидкість зростання об\'єму збільшується: {volume_analysis['volume_acceleration']:.2f}",
                'weight': 0.15
            })
            confidence += 0.15 * min(volume_analysis['volume_acceleration'] / 0.2, 1.0)

        # НОВИЙ Сигнал 8: Значна зміна ціни за 24 години
        if price_analysis.get('price_change_24h', 0) > 50:
            signals.append({
                'name': 'Значна зміна ціни за 24 години',
                'description': f"Ціна зросла на {price_analysis['price_change_24h']:.2f}% за останні 24 години",
                'weight': 0.40
            })
            confidence += 0.40 * min(price_analysis['price_change_24h'] / 100, 1.0)

        # НОВИЙ Сигнал 9: Виявлено dump фазу після pump
        if price_analysis.get('dump_phase', False):
            signals.append({
                'name': 'Dump фаза після pump',
                'description': f"Ціна знизилась на {abs(price_analysis['distance_from_high']):.2f}% від нещодавнього піку",
                'weight': 0.35
            })
            confidence += 0.35 * min(abs(price_analysis['distance_from_high']) / 30, 1.0)

        # Формування результату
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'probability_score': min(confidence, 1.0),
            'signals': signals,
            'raw_data': {
                'volume': volume_analysis,
                'price': price_analysis,
                'order_book': order_book_analysis,
                'social': social_data,
                'time_pattern': time_pattern_data,
                'correlation': correlation_data
            },
            'recommendation': self._get_recommendation(confidence)
        }

        # Зберігаємо сповіщення, якщо рівень впевненості перевищує поріг
        if confidence > self.alert_threshold:
            self.alerts.append(result)

        # Зберігаємо результат в історії для подальшого аналізу
        coin_name = symbol.split('/')[0]
        if coin_name not in self.coin_history:
            self.coin_history[coin_name] = []

        # Зберігаємо скорочений результат в історії
        historical_entry = {
            'timestamp': datetime.now(),
            'probability_score': result['probability_score'],
            'price': price_analysis.get('price_change_1h', 0),
            'volume': volume_analysis.get('percent_change', 0)
        }
        self.coin_history[coin_name].append(historical_entry)

        # Обмежуємо розмір історії
        if len(self.coin_history[coin_name]) > 100:
            self.coin_history[coin_name] = self.coin_history[coin_name][-100:]

        return result

    def _get_recommendation(self, confidence):
        """
        Формування рекомендацій на основі рівня впевненості

        :param confidence: Рівень впевненості (0 до 1)
        :return: Рядок з рекомендацією
        """
        if confidence > 0.8:
            return "Висока ймовірність pump-and-dump схеми. Рекомендується утриматись від торгівлі цією монетою в даний момент."
        elif confidence > 0.6:
            return "Підвищена підозріла активність. Торгівля пов'язана з високим ризиком."
        elif confidence > 0.4:
            return "Виявлено деякі ознаки незвичайної активності. Слід бути обережним."
        elif confidence > 0.2:
            return "Низький рівень підозрілої активності. Звичайний моніторинг."
        else:
            return "Нормальна ринкова активність. Значних аномалій не виявлено."

