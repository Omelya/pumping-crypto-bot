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
        spread = (order_book['asks'][0][0] / order_book['bids'][0][0] - 1) * 100 if order_book['bids'] and order_book[
            'asks'] else 0

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
