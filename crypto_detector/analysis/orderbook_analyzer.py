import pandas as pd


class OrderBookAnalyzer:
    """
    Клас для аналізу книги ордерів криптовалют.
    Виявляє дисбаланси, стіни продажів/купівлі та інші аномалії.
    """

    def __init__(self):
        """
        Ініціалізація аналізатора книги ордерів
        """
        pass

    async def analyze_order_book(self, order_book):
        """
        Аналіз книги ордерів для визначення дисбалансу

        :param order_book: Дані книги ордерів
        :return: Dict з результатами аналізу
        """
        if not order_book or not order_book['bids'] or not order_book['asks']:
            return {'order_book_signal': False, 'buy_sell_ratio': 1.0}

        # Розрахунок загального обсягу на покупку та продаж
        bids_volume = sum(bid[1] for bid in order_book['bids'])
        asks_volume = sum(ask[1] for ask in order_book['asks'])

        # Розрахунок співвідношення
        if asks_volume == 0:
            buy_sell_ratio = float('inf')  # Уникаємо ділення на нуль
        else:
            buy_sell_ratio = bids_volume / asks_volume

        # Розрахунок концентрації ліквідності
        top_bids_volume = sum(bid[1] for bid in order_book['bids'][:5])
        top_asks_volume = sum(ask[1] for ask in order_book['asks'][:5])

        top_concentration = (top_bids_volume / bids_volume if bids_volume > 0 else 0) + \
                            (top_asks_volume / asks_volume if asks_volume > 0 else 0)

        # Розрахунок стіни продажів/купівлі
        has_sell_wall = False
        has_buy_wall = False

        if len(order_book['asks']) >= 2:
            # Перевіряємо стіну продажів (велике скупчення ордерів на продаж)
            top_ask_volume = order_book['asks'][0][1]
            second_ask_volume = order_book['asks'][1][1]
            has_sell_wall = top_ask_volume > second_ask_volume * 5

        if len(order_book['bids']) >= 2:
            # Перевіряємо стіну покупок
            top_bid_volume = order_book['bids'][0][1]
            second_bid_volume = order_book['bids'][1][1]
            has_buy_wall = top_bid_volume > second_bid_volume * 5

        # Аналіз спреду
        spread = (order_book['asks'][0][0] / order_book['bids'][0][0] - 1) * 100 if order_book['bids'] and order_book['asks'] else 0

        return {
            'order_book_signal': buy_sell_ratio > 1.3 or buy_sell_ratio < 0.77 or has_buy_wall or has_sell_wall,
            'buy_sell_ratio': buy_sell_ratio,
            'buy_volume': bids_volume,
            'sell_volume': asks_volume,
            'top_concentration': top_concentration,
            'spread': spread,
            'has_buy_wall': has_buy_wall,
            'has_sell_wall': has_sell_wall
        }

    async def _analyze_historical_orderbook(self, timestamp, data_window):
        """
        Аналіз книги ордерів на основі історичних даних

        :param timestamp: Часова мітка аналізу
        :param data_window: Поточне вікно даних OHLCV
        :return: Результат аналізу книги ордерів
        """
        try:
            # Перевіряємо наявність даних
            if data_window.empty or len(data_window) < 5:
                return {'order_book_signal': False, 'buy_sell_ratio': 1.0}

            # Беремо останню свічку з доступного вікна
            last_candle = data_window.iloc[-1]

            # Оцінюємо потенційний дисбаланс книги ордерів на основі доступних OHLCV даних
            # 1. Використовуємо різницю між High і Close як індикатор тиску продажів
            # 2. Використовуємо різницю між Close і Low як індикатор тиску покупок
            sell_pressure = last_candle['high'] - last_candle['close']
            buy_pressure = last_candle['close'] - last_candle['low']

            # Уникаємо ділення на нуль
            if sell_pressure == 0:
                buy_sell_ratio = 5.0  # Встановлюємо високе значення, якщо тиск продажів відсутній
            else:
                buy_sell_ratio = buy_pressure / sell_pressure

            # Аналізуємо об'єм відносно цінових рухів
            # Високий об'єм при малому русі ціни може свідчити про стіну
            price_range = last_candle['high'] - last_candle['low']
            price_to_volume_ratio = price_range / last_candle['volume'] if last_candle['volume'] > 0 else 0

            # Визначаємо наявність стін на основі структури свічки
            has_buy_wall = False
            has_sell_wall = False

            # Якщо тіло свічки маленьке, але нижній тінь великий - можлива стіна покупок
            if price_range > 0:  # Уникаємо ділення на нуль
                body_to_range_ratio = abs(last_candle['close'] - last_candle['open']) / price_range
                buy_pressure_ratio = buy_pressure / price_range if price_range > 0 else 0
                sell_pressure_ratio = sell_pressure / price_range if price_range > 0 else 0

                if body_to_range_ratio < 0.3 and buy_pressure_ratio > 0.7:
                    has_buy_wall = True

                # Якщо тіло свічки маленьке, але верхній тінь великий - можлива стіна продажів
                if body_to_range_ratio < 0.3 and sell_pressure_ratio > 0.7:
                    has_sell_wall = True

            # Оцінка спреду на основі волатильності за останні N свічок
            volatility = data_window['close'].pct_change().std() * 100  # у відсотках
            estimated_spread = volatility * 0.2  # Приблизна оцінка спреду як частка волатильності

            # Аналіз концентрації об'єму
            # Порівнюємо об'єм поточної свічки з середнім за попередні свічки
            volume_concentration = 1.0
            if len(data_window) > 1 and data_window['volume'].iloc[:-1].mean() > 0:
                volume_concentration = last_candle['volume'] / data_window['volume'].iloc[:-1].mean()

            top_concentration = volume_concentration * 0.5  # Симуляція концентрації верхніх ордерів

            # Додаткові ознаки для історичного аналізу
            # Аналізуємо зміну ціни відразу після вибраної часової точки, якщо дані доступні
            future_price_change = 0

            # Конвертуємо timestamp до відповідного формату
            if isinstance(timestamp, str):
                timestamp_dt = pd.Timestamp(timestamp)
            elif isinstance(timestamp, pd.Timestamp):
                timestamp_dt = timestamp
            else:
                timestamp_dt = pd.Timestamp(timestamp)

            # Перевіряємо, чи є timestamp у індексі
            try:
                # Якщо індекс datetime
                if isinstance(data_window.index, pd.DatetimeIndex):
                    # Знаходимо найближчу часову мітку
                    closest_idx = data_window.index.get_indexer([timestamp_dt], method='nearest')[0]

                    if 0 <= closest_idx < len(data_window) - 1:
                        future_price = data_window['close'].iloc[closest_idx + 1]
                        current_price = data_window['close'].iloc[closest_idx]
                        future_price_change = (future_price / current_price - 1) * 100
                else:
                    # Якщо індекс не datetime, використовуємо останню свічку
                    future_price_change = 0
            except Exception as e:
                # Якщо виникла помилка при пошуку індексу
                future_price_change = 0

            # Визначаємо сигнал дисбалансу
            order_book_signal = (buy_sell_ratio > 1.3 or buy_sell_ratio < 0.77 or
                                 has_buy_wall or has_sell_wall or
                                 volume_concentration > 2.0)

            return {
                'order_book_signal': order_book_signal,
                'buy_sell_ratio': buy_sell_ratio,
                'estimated_buy_volume': buy_pressure * last_candle['volume'],
                'estimated_sell_volume': sell_pressure * last_candle['volume'],
                'top_concentration': top_concentration,
                'spread': estimated_spread,
                'has_buy_wall': has_buy_wall,
                'has_sell_wall': has_sell_wall,
                'volume_concentration': volume_concentration,
                'future_price_change': future_price_change
            }

        except Exception as e:
            print(f"Помилка аналізу історичної книги ордерів: {e}")
            return {'order_book_signal': False, 'buy_sell_ratio': 1.0}
