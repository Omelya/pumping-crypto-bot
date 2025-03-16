import os
import json
import numpy as np
from datetime import datetime

from crypto_detector.data.data_storage import DataStorage


class SignalWeightsManager:
    """
    Клас для управління вагами сигналів для різних категорій токенів.
    Дозволяє оптимізувати, зберігати та завантажувати ваги.
    """

    def __init__(self, storage=None):
        """
        Ініціалізація менеджера ваг сигналів

        :param storage: Система зберігання даних
        """
        self.storage = storage or DataStorage()

        # Дефолтні ваги сигналів
        self.default_weights = {
            'Аномальний обсяг торгів': 0.35,
            'Активна цінова динаміка': 0.25,
            'Дисбаланс книги ордерів': 0.20,
            'Підвищена активність у соціальних мережах': 0.20,
            'Нові лістинги на біржах': 0.10,
            'Підозрілий часовий патерн': 0.15,
            'Корельована активність з іншими монетами': 0.15,
            'Прискорення зростання об\'єму': 0.15
        }

        # Оптимізовані ваги для різних категорій токенів
        self.token_weights = {}

        # Параметри для адаптивного навчання
        self.learning_rate = 0.05  # Швидкість навчання
        self.min_weight = 0.05  # Мінімальна вага сигналу
        self.max_weight = 0.50  # Максимальна вага сигналу
        self.total_max_weight = 1.5  # Максимальна сума ваг

        # Завантаження збережених ваг
        self._load_weights()

    def _load_weights(self):
        """
        Завантаження збережених ваг для різних категорій
        """
        try:
            # Завантаження загальних ваг
            general_weights = self.storage.load_signal_weights()
            if general_weights:
                self.default_weights.update(general_weights)

            # Завантаження ваг для кожної категорії
            categories = ['meme', 'defi', 'l1_blockchain', 'l2_scaling', 'gaming', 'exchange', 'other']
            for category in categories:
                weights = self.storage.load_signal_weights(category)
                if weights:
                    self.token_weights[category] = weights

            print(f"Завантажено ваги для {len(self.token_weights)} категорій токенів")
        except Exception as e:
            print(f"Помилка при завантаженні ваг: {str(e)}")

    def get_weights_for_category(self, category):
        """
        Отримання ваг для конкретної категорії токенів

        :param category: Категорія токена
        :return: Словник з вагами сигналів
        """
        return self.token_weights.get(category, self.default_weights)

    def get_weights_for_token(self, symbol):
        """
        Отримання ваг для конкретного токена

        :param symbol: Символ криптовалюти
        :return: Словник з вагами сигналів
        """
        category = self._get_token_category(symbol)
        return self.get_weights_for_category(category)

    def _get_token_category(self, symbol):
        """
        Визначення категорії токена

        :param symbol: Символ криптовалюти
        :return: Категорія токена
        """
        # Видобуваємо назву монети з символу
        coin_name = symbol.split('/')[0]

        # Мапа категорій токенів
        token_categories = {
            'meme': ['PEPE', 'SHIB', 'DOGE', 'FLOKI', 'CAT', 'CTT', 'WIF'],
            'defi': ['UNI', 'AAVE', 'CAKE', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI'],
            'l1_blockchain': ['ETH', 'SOL', 'ADA', 'AVAX', 'DOT', 'NEAR', 'FTM', 'ATOM'],
            'l2_scaling': ['MATIC', 'ARB', 'OP', 'IMX', 'ZK', 'BASE', 'STX'],
            'gaming': ['AXS', 'SAND', 'MANA', 'ENJ', 'GALA', 'ILV'],
            'exchange': ['BNB', 'CRO', 'FTT', 'KCS', 'LEO', 'OKB']
        }

        # Пошук категорії токена
        for category, tokens in token_categories.items():
            if coin_name in tokens:
                return category

        return 'other'

    def update_weights(self, category, correct_predictions, wrong_predictions):
        """
        Оновлення ваг на основі правильних та неправильних передбачень

        :param category: Категорія токена
        :param correct_predictions: Список правильних передбачень
        :param wrong_predictions: Список неправильних передбачень
        :return: Оновлені ваги
        """
        # Отримання поточних ваг для категорії
        current_weights = self.token_weights.get(category, self.default_weights.copy())

        # Якщо немає помилкових передбачень, нічого оновлювати
        if not wrong_predictions:
            return current_weights

        # Обчислюємо середню активацію сигналів у правильних та помилкових передбаченнях
        signal_importance = {}

        # Для кожного сигналу визначаємо наскільки він був активний у правильних і неправильних передбаченнях
        for signal_name in current_weights.keys():
            correct_activation = sum(1 for e in correct_predictions if signal_name in e.get('signals', {}))
            wrong_activation = sum(1 for e in wrong_predictions if signal_name in e.get('signals', {}))

            correct_ratio = correct_activation / len(correct_predictions) if correct_predictions else 0
            wrong_ratio = wrong_activation / len(wrong_predictions) if wrong_predictions else 0

            # Значення важливості - різниця між активацією в правильних і неправильних передбаченнях
            signal_importance[signal_name] = correct_ratio - wrong_ratio

        # Оновлюємо ваги на основі важливості сигналів
        updated_weights = current_weights.copy()

        for signal_name, importance in signal_importance.items():
            # Збільшуємо або зменшуємо вагу залежно від важливості
            updated_weights[signal_name] = max(
                self.min_weight,
                min(self.max_weight, current_weights[signal_name] + self.learning_rate * importance)
            )

        # Нормалізуємо ваги, щоб їх сума не перевищувала total_max_weight
        weight_sum = sum(updated_weights.values())
        if weight_sum > self.total_max_weight:
            scaling_factor = self.total_max_weight / weight_sum
            updated_weights = {k: v * scaling_factor for k, v in updated_weights.items()}

        # Зберігаємо оновлені ваги
        self.token_weights[category] = updated_weights
        self.storage.save_signal_weights(updated_weights, category)

        return updated_weights

    def optimize_weights(self, training_data, category):
        """
        Оптимізація ваг на основі навчальних даних

        :param training_data: Дані для оптимізації
        :param category: Категорія токена
        :return: Оптимізовані ваги
        """
        # Розділення на правильні та неправильні передбачення
        correct_predictions = []
        wrong_predictions = []

        for entry in training_data:
            predicted_event = entry.get('probability', 0) > 0.5
            actual_event = entry.get('actual_event', False)

            if predicted_event == actual_event:
                correct_predictions.append(entry)
            else:
                wrong_predictions.append(entry)

        # Оновлення ваг
        optimized_weights = self.update_weights(category, correct_predictions, wrong_predictions)

        return optimized_weights

    def evaluate_weights(self, test_data, category, threshold=0.5):
        """
        Оцінка ефективності ваг на тестових даних

        :param test_data: Тестові дані
        :param category: Категорія токена
        :param threshold: Поріг для класифікації
        :return: Метрики ефективності
        """
        # Отримання ваг для категорії
        weights = self.get_weights_for_category(category)

        # Підготовка для розрахунку метрик
        y_true = []
        y_pred = []

        for entry in test_data:
            # Справжнє значення
            y_true.append(1 if entry.get('actual_event', False) else 0)

            # Перерахунок ймовірності з новими вагами
            signals = entry.get('signals', {})
            probability = 0.0

            for signal_name, signal_data in signals.items():
                if signal_name in weights:
                    # Отримання внеску сигналу
                    contribution = signal_data.get('contribution', 0.0)
                    # Нормалізація на поточну вагу
                    current_weight = signal_data.get('weight', 0.0)
                    if current_weight > 0:
                        normalized_contribution = contribution / current_weight
                        # Розрахунок нового внеску з оптимізованою вагою
                        probability += weights[signal_name] * normalized_contribution

            # Передбачення
            y_pred.append(1 if probability > threshold else 0)

        # Розрахунок метрик
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }

        return metrics

    def find_optimal_threshold(self, test_data, category):
        """
        Пошук оптимального порогу для класифікації

        :param test_data: Тестові дані
        :param category: Категорія токена
        :return: Оптимальний поріг та метрики
        """
        # Отримання ваг для категорії
        weights = self.get_weights_for_category(category)

        # Підготовка для розрахунку метрик
        y_true = [1 if entry.get('actual_event', False) else 0 for entry in test_data]
        probabilities = []

        for entry in test_data:
            # Перерахунок ймовірності з новими вагами
            signals = entry.get('signals', {})
            probability = 0.0

            for signal_name, signal_data in signals.items():
                if signal_name in weights:
                    # Отримання внеску сигналу
                    contribution = signal_data.get('contribution', 0.0)
                    # Нормалізація на поточну вагу
                    current_weight = signal_data.get('weight', 0.0)
                    if current_weight > 0:
                        normalized_contribution = contribution / current_weight
                        # Розрахунок нового внеску з оптимізованою вагою
                        probability += weights[signal_name] * normalized_contribution

            probabilities.append(probability)

        # Перебір різних порогів
        from sklearn.metrics import f1_score

        thresholds = np.linspace(0.1, 0.9, 33)  # Від 0.1 до 0.9 з кроком 0.025
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = None

        for threshold in thresholds:
            y_pred = [1 if p > threshold else 0 for p in probabilities]

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }

        return best_threshold, best_metrics

    def save_weights(self):
        """
        Збереження всіх ваг
        """
        # Збереження дефолтних ваг
        self.storage.save_signal_weights(self.default_weights)

        # Збереження ваг для кожної категорії
        for category, weights in self.token_weights.items():
            self.storage.save_signal_weights(weights, category)