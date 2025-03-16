import numpy as np
from collections import deque


class SocialAnalyzer:
    """
    Клас для аналізу соціальної активності навколо криптовалют.
    Відстежує згадки у соціальних мережах та виявляє аномальні сплески активності.

    Примітка: Поточна реалізація використовує симуляцію даних.
    Для реального використання необхідно інтегрувати API соцмереж (Twitter, Reddit, тощо).
    """

    def __init__(self):
        """
        Ініціалізація аналізатора соціальної активності
        """
        self.social_history = {}  # Історія згадок для різних монет

    async def detect_social_media_mentions(self, symbol):
        """
        Аналіз згадок у соціальних мережах

        :param symbol: Символ криптовалюти
        :return: Dict з результатами аналізу
        """
        # Видобуваємо назву монети з символу
        coin_name = symbol.split('/')[0]

        # Збереження історії згадок, якщо раніше не збиралася
        if coin_name not in self.social_history:
            self.social_history[coin_name] = deque(maxlen=24)  # Зберігаємо дані за останні 24 години

            # Заповнюємо початкові дані (в реальному коді тут будуть реальні дані з API)
            base_mentions = np.random.normal(500, 100, 23)
            for mentions in base_mentions:
                self.social_history[coin_name].append(int(mentions))

        # Імітуємо поточне значення (у реальності тут буде запит до API)
        # Для демонстрації додаємо можливе зростання активності
        current_mentions = int(np.random.normal(500, 100))
        if np.random.random() < 0.25:  # 25% шанс симуляції зростання
            current_mentions *= 2.5

        self.social_history[coin_name].append(current_mentions)

        # Розрахунок середнього значення по історії
        avg_mentions = np.mean(list(self.social_history[coin_name])[:-1]) if len(
            self.social_history[coin_name]) > 1 else current_mentions

        # Розрахунок зміни
        percent_change = ((current_mentions / avg_mentions) - 1) * 100 if avg_mentions > 0 else 0

        # Аналіз зростання швидкості згадок
        mentions_history = list(self.social_history[coin_name])
        if len(mentions_history) >= 6:
            recent_growth_rate = (mentions_history[-1] / mentions_history[-2] - 1) if mentions_history[-2] > 0 else 0
            earlier_growth_rate = (mentions_history[-3] / mentions_history[-4] - 1) if mentions_history[-4] > 0 else 0
            growth_acceleration = recent_growth_rate - earlier_growth_rate
        else:
            growth_acceleration = 0

        return {
            'social_signal': percent_change > 40 or growth_acceleration > 0.5,
            'mentions': current_mentions,
            'average_mentions': avg_mentions,
            'percent_change': percent_change,
            'growth_acceleration': growth_acceleration
        }

    def get_coin_social_history(self, coin_name):
        """
        Отримання історії згадок для конкретної монети

        :param coin_name: Назва монети
        :return: Історія згадок або None, якщо історії немає
        """
        return list(self.social_history.get(coin_name, []))

    def clear_history(self):
        """
        Очищення історії соціальної активності
        """
        self.social_history = {}
