import numpy as np
import crypto_detector.config.settings as settings

from sklearn.metrics import precision_score, recall_score

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
        self.default_weights = settings.DEFAULT_SIGNAL_WEIGHTS

        # Оптимізовані ваги для різних категорій токенів
        self.token_weights = {}

        # Параметри для адаптивного навчання
        self.learning_rate = settings.LEARNING_RATE  # Швидкість навчання
        self.min_weight = settings.MIN_WEIGHT  # Мінімальна вага сигналу
        self.max_weight = settings.MAX_WEIGHT  # Максимальна вага сигналу
        self.total_max_weight = settings.TOTAL_MAX_WEIGHT  # Максимальна сума ваг

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
        token_categories = settings.TOKEN_CATEGORIES

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
            try:
                # Підрахунок активацій у правильних передбаченнях
                correct_activation = 0
                for entry in correct_predictions:
                    if 'signals' in entry and isinstance(entry['signals'], dict) and signal_name in entry['signals']:
                        correct_activation += 1

                # Підрахунок активацій у неправильних передбаченнях
                wrong_activation = 0
                for entry in wrong_predictions:
                    if 'signals' in entry and isinstance(entry['signals'], dict) and signal_name in entry['signals']:
                        wrong_activation += 1

                # Розрахунок співвідношень
                correct_ratio = correct_activation / len(correct_predictions) if correct_predictions else 0
                wrong_ratio = wrong_activation / len(wrong_predictions) if wrong_predictions else 0

                # Значення важливості - різниця між активацією в правильних і неправильних передбаченнях
                signal_importance[signal_name] = correct_ratio - wrong_ratio
            except Exception as e:
                print(f"Помилка при обробці сигналу {signal_name}: {str(e)}")
                signal_importance[signal_name] = 0

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
        self.save_weights()

        return updated_weights

    def optimize_weights(self, training_data, category):
        """
        Оптимізація ваг на основі навчальних даних

        :param training_data: Дані для оптимізації
        :param category: Категорія токена
        :return: Оптимізовані ваги
        """
        if not training_data:
            print("Немає даних для оптимізації ваг")
            return self.token_weights.get(category, self.default_weights.copy())

        # Розділення на правильні та неправильні передбачення
        correct_predictions = []
        wrong_predictions = []

        for entry in training_data:
            if not isinstance(entry, dict):
                continue

            probability = entry.get('probability', 0)
            actual_event = entry.get('actual_event')

            # Пропускаємо записи без мітки або ймовірності
            if actual_event is None or probability is None:
                continue

            predicted_event = probability > 0.5
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
        if not test_data:
            print("Немає даних для оцінки ваг")
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'confusion_matrix': [[0, 0], [0, 0]]
            }

        # Отримання ваг для категорії
        weights = self.get_weights_for_category(category)

        # Підготовка для розрахунку метрик
        y_true = []
        y_pred = []

        for entry in test_data:
            if not isinstance(entry, dict):
                continue

            actual_event = entry.get('actual_event')

            # Пропускаємо записи без мітки
            if actual_event is None:
                continue

            # Справжнє значення
            y_true.append(1 if actual_event else 0)

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

        # Перевірка, чи є достатньо даних для розрахунку метрик
        if len(y_true) < 2 or all(y == y_true[0] for y in y_true):
            print("Недостатньо різноманітних даних для розрахунку метрик")
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'confusion_matrix': [[0, 0], [0, 0]]
            }

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
        if not test_data:
            print("Немає даних для пошуку оптимального порогу")
            return 0.5, {}

        # Фільтруємо записи з мітками
        labeled_entries = [e for e in test_data if isinstance(e, dict) and e.get('actual_event') is not None]

        if len(labeled_entries) < 5:
            print("Недостатньо даних для пошуку оптимального порогу")
            return 0.5, {}

        # Підготовка для розрахунку метрик
        y_true = [1 if e.get('actual_event') else 0 for e in labeled_entries]
        probabilities = [e.get('probability', 0) for e in labeled_entries]

        # Перебір різних порогів
        from sklearn.metrics import f1_score

        thresholds = np.linspace(0.1, 0.9, 33)  # Від 0.1 до 0.9 з кроком 0.025
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = None

        for threshold in thresholds:
            y_pred = [1 if p > threshold else 0 for p in probabilities]

            try:
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
            except Exception as e:
                print(f"Помилка при розрахунку метрик для порогу {threshold}: {str(e)}")

        return best_threshold, best_metrics

    def save_weights(self):
        """
        Збереження всіх ваг
        """
        # Збереження дефолтних ваг
        try:
            self.storage.save_signal_weights(self.default_weights)

            # Збереження ваг для кожної категорії
            for category, weights in self.token_weights.items():
                self.storage.save_signal_weights(weights, category)

            print("Ваги сигналів успішно збережено")

        except Exception as e:
            print(f"Помилка при збереженні ваг сигналів: {str(e)}")
