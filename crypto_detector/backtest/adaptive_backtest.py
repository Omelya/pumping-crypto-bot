import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from crypto_detector.backtest.backtest_base import CryptoBacktester
from crypto_detector.core.adaptive_detector import AdaptiveCryptoDetector


class AdaptiveBacktester(CryptoBacktester):
    """
    Розширення CryptoBacktester для використання адаптивного детектора.
    Дозволяє порівнювати результати стандартного та адаптивного підходів.
    """

    def __init__(self, base_detector, data_dir="historical_data"):
        """
        Ініціалізація адаптивного бектестера

        :param base_detector: Базовий детектор
        :param data_dir: Директорія з історичними даними
        """
        super().__init__(base_detector, data_dir)

        # Створюємо адаптивний детектор
        self.adaptive_detector = AdaptiveCryptoDetector(base_detector)

        # Результати для різних стратегій
        self.comparative_results = {}

    async def backtest_with_adaptive_learning(self, symbol, start_date, end_date, min_price_change=5.0,
                                              window_hours=24):
        """
        Бектестинг з адаптивним навчанням

        :param symbol: Символ криптовалюти
        :param start_date: Початкова дата
        :param end_date: Кінцева дата
        :param min_price_change: Мінімальна зміна ціни для позначення події
        :param window_hours: Вікно для аналізу в годинах
        :return: Результати бектестингу
        """
        # Запускаємо стандартний бектестинг для порівняння
        standard_results = await super().backtest_algorithm(symbol, start_date, end_date, min_price_change,
                                                            window_hours)

        if standard_results is None:
            return None

        # Завантаження даних (вже завантажені в стандартному бектестингу)
        data_file = os.path.join(
            self.data_dir,
            symbol.replace('/', '_'),
            f"5m_{start_date}_{end_date}.csv"
        )

        historical_data = self.load_historical_data(data_file)
        if historical_data is None:
            return None

        # Генерація тестових подій
        events_df = self.generate_test_events(historical_data, min_price_change, window=int(window_hours * 12))

        if events_df.empty:
            print("Не знайдено тестових подій")
            return None

        # Тестові точки
        test_timestamps = []

        # Додавання подій як тестових точок
        for event_time in events_df.index:
            test_timestamps.append(event_time)

        # Додавання випадкових точок, які не є подіями (з кращим балансом)
        non_event_times = historical_data.index.difference(events_df.index)
        if len(non_event_times) > len(events_df):
            # Для кращого балансу класів
            balance_ratio = 1.5
            random_non_events = np.random.choice(non_event_times,
                                                 size=min(int(len(events_df) * balance_ratio), len(non_event_times)),
                                                 replace=False)
            test_timestamps.extend(random_non_events)

        # Сортування часових міток для послідовного аналізу
        test_timestamps.sort()

        # Бектестинг на кожній тестовій точці з адаптивним навчанням
        y_true = []
        y_pred = []
        y_pred_adaptive = []  # Окремо для адаптивного підходу
        predictions = []

        # Динамічний поріг для класифікації
        standard_threshold = self.get_prediction_threshold(symbol)
        adaptive_threshold = self.adaptive_detector.get_optimized_threshold(symbol)

        print(f"Порівняння порогів для {symbol}: стандартний {standard_threshold}, адаптивний {adaptive_threshold}")

        # Запуск адаптивного бектестингу
        for i, timestamp in enumerate(test_timestamps):
            # Визначення чи є ця точка подією
            is_event = 1 if timestamp in events_df.index else 0
            y_true.append(is_event)

            # Підготовка даних для аналізу
            cutoff_time = timestamp
            data_window = historical_data[:cutoff_time].tail(int(window_hours * 12))

            # Перевірка наявності достатньої кількості даних
            if len(data_window) < window_hours:
                print(f"Недостатньо даних для аналізу на {timestamp}")
                y_pred.append(0)
                y_pred_adaptive.append(0)
                continue

            # Виклик стандартного методу аналізу
            standard_result = await self._analyze_historical_point(self.detector, symbol, data_window, timestamp)

            # Виклик адаптивного методу аналізу
            adaptive_result = await self.adaptive_detector.analyze_historical_point(symbol, data_window, timestamp)

            # Визначення передбачення за стандартним порогом
            standard_prediction = 1 if standard_result['probability_score'] > standard_threshold else 0
            y_pred.append(standard_prediction)

            # Визначення передбачення за адаптивним порогом
            adaptive_prediction = 1 if adaptive_result['probability_score'] > adaptive_threshold else 0
            y_pred_adaptive.append(adaptive_prediction)

            # Збереження деталей
            predictions.append({
                'timestamp': pd.Timestamp(timestamp).isoformat(),
                'is_event': is_event,
                'standard_prediction': standard_prediction,
                'adaptive_prediction': adaptive_prediction,
                'standard_probability': standard_result['probability_score'],
                'adaptive_probability': adaptive_result['probability_score'],
                'signals': standard_result['signals'],
                'adaptive_signals': adaptive_result['signals']
            })

            # Якщо це не перші точки, надаємо зворотний зв'язок для адаптивного навчання
            if i >= 10:  # Даємо деякий початковий набір даних
                # Перевіряємо, чи вистачає даних для навчання
                if len(predictions) >= 10:
                    # Обмежуємо індекс, щоб не вийти за межі масиву
                    feedback_count = min(10, i)  # Не більше 10 зразків і не більше ніж поточний індекс
                    for idx in range(max(0, len(predictions) - feedback_count), len(predictions)):
                        prev_prediction = predictions[idx]
                        self.adaptive_detector.provide_feedback(
                            symbol,
                            prev_prediction['timestamp'],
                            prev_prediction['is_event'] == 1
                        )

            print(f"Аналіз на {timestamp}: Подія={is_event}, Стандартний={standard_prediction}, " +
                  f"Адаптивний={adaptive_prediction}, " +
                  f"Ймов.станд={standard_result['probability_score']:.2f}, " +
                  f"Ймов.адапт={adaptive_result['probability_score']:.2f}")

        # Розрахунок метрик для обох підходів
        if len(y_true) > 0 and len(y_pred) > 0 and len(y_pred_adaptive) > 0:
            # Метрики для стандартного підходу
            std_precision = precision_score(y_true, y_pred, zero_division=0)
            std_recall = recall_score(y_true, y_pred, zero_division=0)
            std_f1 = f1_score(y_true, y_pred, zero_division=0)
            std_cm = confusion_matrix(y_true, y_pred)

            # Метрики для адаптивного підходу
            adp_precision = precision_score(y_true, y_pred_adaptive, zero_division=0)
            adp_recall = recall_score(y_true, y_pred_adaptive, zero_division=0)
            adp_f1 = f1_score(y_true, y_pred_adaptive, zero_division=0)
            adp_cm = confusion_matrix(y_true, y_pred_adaptive)

            # Порівняння результатів
            print("\nПорівняння результатів бектестингу:")
            print(f"{'Метрика':<15} {'Стандартний':<15} {'Адаптивний':<15} {'Покращення':<15}")
            print("-" * 60)
            print(f"{'Precision':<15} {std_precision:.4f}{'':<9} {adp_precision:.4f}{'':<9} " +
                  f"{((adp_precision - std_precision) / std_precision * 100) if std_precision > 0 else 0:.2f}%")
            print(f"{'Recall':<15} {std_recall:.4f}{'':<9} {adp_recall:.4f}{'':<9} " +
                  f"{((adp_recall - std_recall) / std_recall * 100) if std_recall > 0 else 0:.2f}%")
            print(f"{'F1 Score':<15} {std_f1:.4f}{'':<9} {adp_f1:.4f}{'':<9} " +
                  f"{((adp_f1 - std_f1) / std_f1 * 100) if std_f1 > 0 else 0:.2f}%")

            # Збереження результатів
            adaptive_results = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'standard': {
                    'precision': std_precision,
                    'recall': std_recall,
                    'f1_score': std_f1,
                    'confusion_matrix': std_cm.tolist()
                },
                'adaptive': {
                    'precision': adp_precision,
                    'recall': adp_recall,
                    'f1_score': adp_f1,
                    'confusion_matrix': adp_cm.tolist()
                },
                'improvement': {
                    'precision': ((adp_precision - std_precision) / std_precision * 100) if std_precision > 0 else 0,
                    'recall': ((adp_recall - std_recall) / std_recall * 100) if std_recall > 0 else 0,
                    'f1_score': ((adp_f1 - std_f1) / std_f1 * 100) if std_f1 > 0 else 0
                },
                'predictions': predictions
            }

            self.comparative_results[symbol] = adaptive_results

            # Збереження ваг сигналів для подальшого аналізу
            category = self.adaptive_detector._get_token_category(symbol)
            optimized_weights = self.adaptive_detector.get_weights_for_token(symbol)

            print(f"\nОптимізовані ваги для категорії '{category}':")
            for signal, weight in optimized_weights.items():
                print(f"{signal:<45}: {weight:.3f}")

            # Підготовка даних для ML навчання
            ml_data = pd.DataFrame(predictions)
            ml_data_file = f"{symbol.replace('/', '_')}_adaptive_ml_data.csv"
            ml_data.to_csv(ml_data_file, index=False)
            print(f"Дані для машинного навчання збережені у {ml_data_file}")

            return adaptive_results
        else:
            print("Не вдалося виконати адаптивний бектестинг через недостатню кількість даних")
            return None

    def save_comparative_results(self, filename="adaptive_backtest_results.json"):
        """
        Збереження результатів порівняльного бектестингу у файл

        :param filename: Ім'я файлу для збереження
        """
        with open(filename, 'w') as f:
            json.dump(self.comparative_results, f, indent=4, default=str)
        print(f"Результати порівняльного бектестингу збережено у {filename}")

    def visualize_comparative_results(self, symbol):
        """
        Візуалізація порівняльних результатів бектестингу

        :param symbol: Символ криптовалюти
        """
        if symbol not in self.comparative_results:
            print(f"Немає порівняльних результатів бектестингу для {symbol}")
            return

        results = self.comparative_results[symbol]

        # Підготовка даних для візуалізації
        predictions_df = pd.DataFrame(results['predictions'])
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df.set_index('timestamp', inplace=True)
        predictions_df.sort_index(inplace=True)

        # Створення візуалізацій
        plt.figure(figsize=(20, 15))

        # Графік 1: Порівняння ймовірностей
        plt.subplot(3, 2, 1)
        plt.plot(predictions_df.index, predictions_df['standard_probability'], 'b-', label='Стандартний алгоритм')
        plt.plot(predictions_df.index, predictions_df['adaptive_probability'], 'r-', label='Адаптивний алгоритм')

        # Додаємо маркери для реальних подій
        event_times = predictions_df[predictions_df['is_event'] == 1].index

        # Перевіряємо наявність подій для уникнення помилок
        if len(event_times) > 0:
            event_std_probs = predictions_df.loc[event_times, 'standard_probability']
            event_adp_probs = predictions_df.loc[event_times, 'adaptive_probability']

            # Перевіряємо, що розміри масивів співпадають
            if len(event_times) == len(event_std_probs) and len(event_times) == len(event_adp_probs):
                plt.scatter(event_times, event_std_probs, c='b', marker='o', s=100, alpha=0.5)
                plt.scatter(event_times, event_adp_probs, c='r', marker='o', s=100, alpha=0.5)
            else:
                print(f"Попередження: розміри масивів не співпадають!")

        # Додаємо лінію порогу
        plt.axhline(y=self.get_prediction_threshold(symbol), color='b', linestyle='--', alpha=0.7,
                    label=f'Стандартний поріг')
        plt.axhline(y=self.adaptive_detector.get_optimized_threshold(symbol), color='r', linestyle='--', alpha=0.7,
                    label=f'Адаптивний поріг')

        plt.title(f'Порівняння ймовірностей pump-and-dump для {symbol}')
        plt.xlabel('Дата')
        plt.ylabel('Ймовірність')
        plt.legend()
        plt.grid(True)

        # Графік 2: Гістограми розподілу ймовірностей
        plt.subplot(3, 2, 2)

        # Розділяємо на події та не-події
        events = predictions_df[predictions_df['is_event'] == 1]
        non_events = predictions_df[predictions_df['is_event'] == 0]

        # Гістограми для подій
        plt.hist(events['standard_probability'], bins=15, alpha=0.5, color='blue', label='Стандартний (події)')
        plt.hist(events['adaptive_probability'], bins=15, alpha=0.5, color='red', label='Адаптивний (події)')

        # Гістограми для не-подій
        plt.hist(non_events['standard_probability'], bins=15, alpha=0.3, color='cyan', label='Стандартний (не-події)')
        plt.hist(non_events['adaptive_probability'], bins=15, alpha=0.3, color='orange', label='Адаптивний (не-події)')

        plt.title('Розподіл ймовірностей для подій та не-подій')
        plt.xlabel('Ймовірність')
        plt.ylabel('Кількість')
        plt.legend()
        plt.grid(True)

        # Графік 3: Помилкові спрацювання і пропуски
        plt.subplot(3, 2, 3)

        # Створюємо таблицю плутанини для візуалізації
        standard_cm = np.array(results['standard']['confusion_matrix'])
        adaptive_cm = np.array(results['adaptive']['confusion_matrix'])

        labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        std_values = [standard_cm[0, 0], standard_cm[0, 1], standard_cm[1, 0], standard_cm[1, 1]]
        adp_values = [adaptive_cm[0, 0], adaptive_cm[0, 1], adaptive_cm[1, 0], adaptive_cm[1, 1]]

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, std_values, width, label='Стандартний')
        plt.bar(x + width / 2, adp_values, width, label='Адаптивний')

        plt.title('Порівняння компонентів матриці плутанини')
        plt.xticks(x, labels)
        plt.ylabel('Кількість')
        plt.legend()
        plt.grid(True)

        # Графік 4: Порівняння метрик продуктивності
        plt.subplot(3, 2, 4)

        metrics = ['Precision', 'Recall', 'F1 Score']
        std_metrics = [results['standard']['precision'], results['standard']['recall'], results['standard']['f1_score']]
        adp_metrics = [results['adaptive']['precision'], results['adaptive']['recall'], results['adaptive']['f1_score']]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width / 2, std_metrics, width, label='Стандартний')
        plt.bar(x + width / 2, adp_metrics, width, label='Адаптивний')

        plt.title('Порівняння метрик ефективності')
        plt.xticks(x, metrics)
        plt.ylabel('Значення')
        plt.legend()
        plt.grid(True)

        # Графік 5: Часова динаміка правильних та помилкових спрацювань
        plt.subplot(3, 2, 5)

        # Створюємо нові поля для правильності передбачень
        predictions_df['std_correct'] = (predictions_df['standard_prediction'] == predictions_df['is_event']).astype(
            int)
        predictions_df['adp_correct'] = (predictions_df['adaptive_prediction'] == predictions_df['is_event']).astype(
            int)

        # Обчислюємо ковзне середнє для точності передбачень
        window_size = 10
        predictions_df['std_accuracy_ma'] = predictions_df['std_correct'].rolling(window=window_size).mean()
        predictions_df['adp_accuracy_ma'] = predictions_df['adp_correct'].rolling(window=window_size).mean()

        # Будуємо графік
        plt.plot(predictions_df.index, predictions_df['std_accuracy_ma'], 'b-', label='Стандартний (ковзна точність)')
        plt.plot(predictions_df.index, predictions_df['adp_accuracy_ma'], 'r-', label='Адаптивний (ковзна точність)')

        plt.title(f'Динаміка точності передбачень (ковзне вікно {window_size})')
        plt.xlabel('Дата')
        plt.ylabel('Точність (частка правильних)')
        plt.legend()
        plt.grid(True)

        # Графік 6: Порівняння ваг сигналів
        plt.subplot(3, 2, 6)

        # Отримуємо ваги сигналів
        category = self.adaptive_detector._get_token_category(symbol)
        optimized_weights = self.adaptive_detector.get_weights_for_token(symbol)
        default_weights = self.adaptive_detector.default_weights

        # Сортування за назвами для відповідності
        signals = sorted(optimized_weights.keys())
        opt_weights = [optimized_weights[s] for s in signals]
        def_weights = [default_weights.get(s, 0) for s in signals]

        x = np.arange(len(signals))
        width = 0.35

        plt.bar(x - width / 2, def_weights, width, label='Початкові ваги')
        plt.bar(x + width / 2, opt_weights, width, label='Оптимізовані ваги')

        plt.title(f'Порівняння ваг сигналів для категорії {category}')
        plt.xticks(x, signals, rotation=45, ha='right')
        plt.ylabel('Вага')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{symbol.replace('/', '_')}_comparative_results.png")
        plt.show()

        print(f"Візуалізація збережена як {symbol.replace('/', '_')}_comparative_results.png")
