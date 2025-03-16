import numpy as np
from datetime import datetime


class CorrelationAnalyzer:
    """
    Клас для аналізу кореляції між різними криптовалютами.
    Виявляє скоординовані pump-and-dump атаки, які часто охоплюють кілька монет одночасно.

    Примітка: Поточна реалізація використовує симуляцію даних.
    Для реального використання необхідно накопичувати та аналізувати реальні дані руху цін.
    """

    def __init__(self):
        """
        Ініціалізація аналізатора кореляції
        """
        # Зберігання історії для всіх монет
        self.market_data = {}

    async def analyze_market_correlation(self, symbol, all_symbols):
        """
        Аналіз кореляції з іншими маркетами для виявлення скоординованих pump-and-dump атак

        :param symbol: Поточний символ для аналізу
        :param all_symbols: Список всіх символів для порівняння
        :return: Dict з результатами аналізу
        """
        # Отримуємо базову валюту (наприклад, 'BTC' з 'BTC/USDT')
        base_currency = symbol.split('/')[0]

        # Ми аналізуємо тільки альткоїни (не BTC, ETH, BNB тощо)
        major_coins = {'BTC', 'ETH', 'BNB', 'USDT', 'USDC', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE'}
        if base_currency in major_coins:
            return {'correlation_signal': False}

        # Для демонстрації симулюємо корельовані дані
        pump_correlation = np.random.random() < 0.15  # 15% шанс знайти скорельований памп

        # У реальній імплементації тут має бути:
        # 1. Збір даних про рух цін всіх монет
        # 2. Розрахунок кореляції між різними монетами
        # 3. Кластеризація для виявлення груп монет з синхронним рухом цін
        # 4. Визначення чи даний рух є природнім чи підозрілим

        correlated_coins = []
        if pump_correlation:
            # Симуляція кількох корельованих монет
            correlated_coins = [s for s in all_symbols if s != symbol and np.random.random() < 0.3][:3]

        return {
            'correlation_signal': pump_correlation,
            'correlated_coins': correlated_coins,
            'correlation_type': 'pump_group' if pump_correlation else 'normal'
        }

    def add_market_data(self, symbol, price, volume, timestamp=None):
        """
        Додавання даних про ринкову активність для подальшого аналізу

        :param symbol: Символ криптовалюти
        :param price: Ціна
        :param volume: Об'єм торгів
        :param timestamp: Часова мітка (за замовчуванням - поточний час)
        """
        if timestamp is None:
            timestamp = datetime.now()

        if symbol not in self.market_data:
            self.market_data[symbol] = []

        self.market_data[symbol].append({
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        })

        # Обмежуємо розмір історії для економії пам'яті
        if len(self.market_data[symbol]) > 1000:
            self.market_data[symbol] = self.market_data[symbol][-1000:]

    def get_correlated_groups(self, threshold=0.8):
        """
        Отримання груп корельованих активів

        :param threshold: Поріг кореляції для об'єднання в групу
        :return: Список груп корельованих активів
        """
        # У реальній імплементації тут має бути розрахунок кореляційної матриці
        # та кластеризація на основі значень кореляції

        # Заглушка для демонстрації
        groups = []
        unique_symbols = list(self.market_data.keys())

        # Генеруємо випадкові групи
        remaining = unique_symbols.copy()
        while remaining:
            group_size = min(np.random.randint(2, 5), len(remaining))
            group = [remaining.pop(0) for _ in range(group_size)]
            if len(group) >= 2:  # Потрібно щонайменше 2 елементи для групи
                groups.append(group)

        return groups
