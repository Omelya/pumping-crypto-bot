import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            'currency',
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

            # Створюємо директорії для збереження даних ML
            ml_data_dir = os.path.join(self.data_dir, "ml_data")
            if not os.path.exists(ml_data_dir):
                os.makedirs(ml_data_dir)

            # Підготовка даних для ML навчання
            ml_data = pd.DataFrame(predictions)
            ml_data_file = os.path.join(ml_data_dir, f"{symbol.replace('/', '_')}_adaptive_ml_data.csv")
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

        # Створення директорії для графіків, якщо вона не існує
        charts_dir = os.path.join(self.data_dir, "charts")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

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

        # Збереження графіка у вказаній директорії
        chart_file = os.path.join(charts_dir, f"{symbol.replace('/', '_')}_comparative_results.png")
        plt.savefig(chart_file)
        plt.close()  # Закриваємо фігуру для звільнення пам'яті

        print(f"Візуалізацію збережено як {chart_file}")

    def train_all_ml_models(self):
        """
        Навчання ML моделей для всіх категорій на основі даних бектестингу
        """
        print("\nЗапуск навчання ML моделей на основі результатів бектестингу...")

        # Перевірка наявності директорії для ML даних
        ml_data_dir = os.path.join(self.data_dir, "ml_data")
        if not os.path.exists(ml_data_dir):
            print(f"Директорія {ml_data_dir} не існує. Неможливо навчити моделі.")
            return

        # Отримання списку всіх CSV файлів у директорії ML даних
        ml_files = [f for f in os.listdir(ml_data_dir) if f.endswith('_adaptive_ml_data.csv')]

        if not ml_files:
            print("Не знайдено файлів з даними для ML навчання.")
            return

        print(f"Знайдено {len(ml_files)} файлів з даними для навчання.")

        # Для кожного файлу готуємо дані для відповідного символу/категорії
        for ml_file in ml_files:
            # Отримання символу з імені файлу
            symbol = ml_file.replace('_adaptive_ml_data.csv', '').replace('_', '/')

            # Визначення категорії токена
            category = self.adaptive_detector._get_token_category(symbol)

            print(f"\nПідготовка навчання моделі для {symbol} (категорія: {category})...")

            # Повний шлях до файлу
            ml_data_path = os.path.join(ml_data_dir, ml_file)

            try:
                # Навчання моделі для цієї категорії
                self._train_model_for_category(ml_data_path, category, symbol)
            except Exception as e:
                print(f"Помилка при навчанні моделі для {symbol}: {str(e)}")

        print("\nЗавершення навчання ML моделей. Збереження результатів...")

        # Збереження всіх навчених моделей і ваг
        if hasattr(self.adaptive_detector, '_save_weights'):
            self.adaptive_detector._save_weights()

        print("Навчання ML моделей завершено успішно.")

    def _train_model_for_category(self, ml_data_path, category, symbol):
        """
        Навчання ML моделі для конкретної категорії

        :param ml_data_path: Шлях до файлу з даними
        :param category: Категорія токена
        :param symbol: Символ криптовалюти
        """
        # Перевірка наявності MLTrainer в адаптивному детекторі
        if not hasattr(self.adaptive_detector, 'ml_trainer'):
            print("MLTrainer не ініціалізовано в адаптивному детекторі.")
            return

        try:
            # Підготовка даних
            raw_data = self._load_and_preprocess_data(ml_data_path)
            if raw_data is None:
                return

            # Підготовка фіч для навчання моделі
            X, y = self._prepare_features(raw_data, ml_data_path)
            if X is None or len(X) < 30:
                print(
                    f"Недостатньо даних для навчання моделі: {len(X) if X is not None else 0} зразків. Потрібно мінімум 30.")
                return

            # Оптимізація гіперпараметрів
            best_params = self._optimize_hyperparameters(X, y, category)

            # Навчання моделі з оптимальними параметрами
            model, model_info = self._train_and_save_model(X, y, category, best_params)
            if model is None:
                return

            # Візуалізація результатів навчання
            charts_dir = self._create_charts_directory()
            self._visualize_and_save_feature_importance(model_info, category, symbol, charts_dir)

            # Оцінка моделі через крос-валідацію
            self._perform_cross_validation(X, y, category, best_params, symbol, charts_dir)

            # Виведення метрик моделі
            self._log_model_metrics(model_info, category)

            # Пошук оптимального порогу та оновлення ваг
            self._optimize_threshold_and_update_weights(model, X, y, category, symbol, charts_dir, raw_data)

        except Exception as e:
            print(f"Загальна помилка при навчанні моделі для {category}: {str(e)}")

    def _load_and_preprocess_data(self, ml_data_path):
        """
        Завантаження та попередня обробка даних

        :param ml_data_path: Шлях до файлу з даними
        :return: Підготовлені дані або None у випадку помилки
        """
        try:
            print(f"\nДіагностика даних у файлі {ml_data_path}")
            raw_data = pd.read_csv(ml_data_path)
            print(f"Розмір даних: {raw_data.shape}")
            print(f"Колонки: {raw_data.columns.tolist()}")

            # Попередня обробка даних
            preprocessed_file = ml_data_path.replace('.csv', '_preprocessed.csv')
            raw_data.to_csv(preprocessed_file, index=False)
            print(f"Збережено дані для навчання у {preprocessed_file}")

            return raw_data
        except Exception as e:
            print(f"Помилка при завантаженні та обробці даних: {str(e)}")
            return None

    def _prepare_features(self, raw_data, ml_data_path):
        """
        Підготовка фіч для навчання моделі

        :param raw_data: Завантажені дані
        :param ml_data_path: Шлях до файлу з даними
        :return: Кортеж (X, y) підготовлених фіч та міток або (None, None) у випадку помилки
        """
        try:
            preprocessed_file = ml_data_path.replace('.csv', '_preprocessed.csv')
            print("Передаємо дані в MLTrainer для обробки...")
            X, y = self.adaptive_detector.ml_trainer.prepare_features(preprocessed_file)
            print(f"Підготовлено {len(X)} зразків для навчання.")
            return X, y
        except Exception as e:
            print(f"Помилка при підготовці фіч: {str(e)}")
            return None, None

    def _optimize_hyperparameters(self, X, y, category):
        """
        Оптимізація гіперпараметрів для моделі

        :param X: Підготовлені фічі
        :param y: Мітки класів
        :param category: Категорія токена
        :return: Словник оптимальних гіперпараметрів
        """
        print(f"Оптимізація гіперпараметрів для категорії {category}...")
        try:
            best_params, _ = self.adaptive_detector.ml_trainer.optimize_hyperparameters(
                X, y, model_type='gradient_boosting', category=category
            )
            print(f"Оптимальні гіперпараметри для {category}: {best_params}")
            return best_params
        except Exception as e:
            print(f"Помилка при оптимізації гіперпараметрів: {str(e)}")
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3
            }
            print(f"Використовуємо стандартні гіперпараметри: {default_params}")
            return default_params

    def _train_and_save_model(self, X, y, category, best_params):
        """
        Навчання моделі з оптимальними параметрами та збереження

        :param X: Підготовлені фічі
        :param y: Мітки класів
        :param category: Категорія токена
        :param best_params: Оптимальні гіперпараметри
        :return: Кортеж (модель, інформація про модель) або (None, None) у випадку помилки
        """
        print(f"Навчання моделі для {category}...")
        try:
            model, model_info = self.adaptive_detector.ml_trainer.train_model(
                X, y, model_type='gradient_boosting', category=category, **best_params
            )

            # Збереження моделі в адаптивному детекторі
            self.adaptive_detector.ml_models[category] = model

            # Збереження моделі на диск
            if hasattr(self.adaptive_detector, '_save_ml_model'):
                self.adaptive_detector._save_ml_model(category)

            return model, model_info
        except Exception as e:
            print(f"Помилка при навчанні моделі для {category}: {str(e)}")
            return None, None

    def _create_charts_directory(self):
        """
        Створення директорії для графіків

        :return: Шлях до директорії з графіками
        """
        charts_dir = os.path.join(self.data_dir, "charts", "ml_models")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        return charts_dir

    def _visualize_and_save_feature_importance(self, model_info, category, symbol, charts_dir):
        """
        Візуалізація та збереження графіка важливості фічей

        :param model_info: Інформація про навчену модель
        :param category: Категорія токена
        :param symbol: Символ криптовалюти
        :param charts_dir: Шлях до директорії з графіками
        """
        try:
            import matplotlib.pyplot as plt

            # Візуалізація важливості фічей
            self.adaptive_detector.ml_trainer.visualize_feature_importance(
                model_info['feature_importance'],
                title=f"Важливість ознак для категорії {category}",
                top_n=20
            )

            # Збереження графіка
            chart_path = os.path.join(charts_dir, f"{symbol.replace('/', '_')}_feature_importance.png")
            plt.savefig(chart_path)
            plt.close()

            print(f"Графік важливості ознак збережено у {chart_path}")
        except Exception as e:
            print(f"Помилка при створенні графіка важливості ознак: {str(e)}")

    def _perform_cross_validation(self, X, y, category, best_params, symbol, charts_dir):
        """
        Виконання крос-валідації та збереження звіту

        :param X: Підготовлені фічі
        :param y: Мітки класів
        :param category: Категорія токена
        :param best_params: Оптимальні гіперпараметри
        :param symbol: Символ криптовалюти
        :param charts_dir: Шлях до директорії з графіками
        """
        try:
            cv_report, _ = self.adaptive_detector.ml_trainer.cross_validate_and_report(
                X, y, model_type='gradient_boosting', category=category, **best_params
            )

            # Збереження звіту про крос-валідацію
            report_file = os.path.join(charts_dir, f"{symbol.replace('/', '_')}_cv_report.json")
            import json
            with open(report_file, 'w') as f:
                json.dump(cv_report, f, indent=4, default=str)

            print(f"Звіт про крос-валідацію збережено у {report_file}")
        except Exception as e:
            print(f"Помилка при крос-валідації: {str(e)}")

    def _log_model_metrics(self, model_info, category):
        """
        Виведення метрик навченої моделі

        :param model_info: Інформація про навчену модель
        :param category: Категорія токена
        """
        print(f"Метрики моделі для {category}:")
        print(f"  Precision: {model_info['metrics']['precision']:.4f}")
        print(f"  Recall: {model_info['metrics']['recall']:.4f}")
        print(f"  F1 Score: {model_info['metrics']['f1_score']:.4f}")

    def _optimize_threshold_and_update_weights(self, model, X, y, category, symbol, charts_dir, raw_data):
        """
        Пошук оптимального порогу та оновлення ваг сигналів

        :param model: Навчена модель
        :param X: Підготовлені фічі
        :param y: Мітки класів
        :param category: Категорія токена
        :param symbol: Символ криптовалюти
        :param charts_dir: Шлях до директорії з графіками
        :param raw_data: Початкові дані
        """
        try:
            # Пошук оптимального порогу
            optimal_threshold = self._find_optimal_threshold(model, X, y, category, symbol, charts_dir)

            # Підготовка даних для оновлення ваг
            correct_predictions, wrong_predictions = self._prepare_data_for_weights_update(
                model, X, y, raw_data, optimal_threshold
            )

            # Оновлення ваг сигналів
            self._update_signal_weights(category, correct_predictions, wrong_predictions)
        except Exception as e:
            print(f"Помилка при оптимізації порогу та оновленні ваг: {str(e)}")

    def _find_optimal_threshold(self, model, X, y, category, symbol, charts_dir):
        """
        Пошук оптимального порогу для класифікації

        :param model: Навчена модель
        :param X: Підготовлені фічі
        :param y: Мітки класів
        :param category: Категорія токена
        :param symbol: Символ криптовалюти
        :param charts_dir: Шлях до директорії з графіками
        :return: Оптимальний поріг
        """
        try:
            import matplotlib.pyplot as plt

            y_prob = model.predict_proba(X)[:, 1]
            optimal_threshold, _ = self.adaptive_detector.ml_trainer.plot_threshold_optimization(
                y, y_prob, metric='f1', title=f"Оптимізація порогу для {category}"
            )

            # Збереження графіка оптимізації порогу
            chart_path = os.path.join(charts_dir, f"{symbol.replace('/', '_')}_threshold_optimization.png")
            plt.savefig(chart_path)
            plt.close()

            print(f"Оптимальний поріг для {category}: {optimal_threshold:.4f}")
            return optimal_threshold
        except Exception as e:
            print(f"Помилка при оптимізації порогу: {str(e)}")
            return 0.5  # Повертаємо стандартний поріг

    def _prepare_data_for_weights_update(self, model, X, y, raw_data, threshold=0.5):
        """
        Підготовка даних для оновлення ваг сигналів

        :param model: Навчена модель
        :param X: Підготовлені фічі
        :param y: Мітки класів
        :param raw_data: Початкові дані
        :param threshold: Поріг для класифікації
        :return: Кортеж (правильні передбачення, неправильні передбачення)
        """
        print(f"Підготовка даних для оновлення ваг сигналів...")
        try:
            y_pred_prob = model.predict_proba(X)[:, 1]

            # Переконаємось, що y_pred_prob - це numpy масив
            if not isinstance(y_pred_prob, np.ndarray):
                y_pred_prob = np.array(y_pred_prob)

            # Отримуємо передбачення моделі
            y_pred = (y_pred_prob > threshold).astype(int)

            # Створюємо дані для оновлення ваг
            labeled_entries = []
            for i, (features, true_label, pred_prob) in enumerate(zip(X, y, y_pred_prob)):
                # Завантажуємо оригінальний рядок з даних
                orig_row = raw_data.iloc[i].to_dict() if i < len(raw_data) else {}

                entry = {
                    'probability': pred_prob,
                    'actual_event': bool(true_label),
                    'signals': orig_row.get('signals', {})
                }
                labeled_entries.append(entry)

            # Розділення на правильні та неправильні передбачення
            correct_predictions = [entry for i, entry in enumerate(labeled_entries) if y_pred[i] == y[i]]
            wrong_predictions = [entry for i, entry in enumerate(labeled_entries) if y_pred[i] != y[i]]

            print(
                f"Підготовлено {len(correct_predictions)} правильних та {len(wrong_predictions)} неправильних передбачень")
            return correct_predictions, wrong_predictions
        except Exception as e:
            print(f"Помилка при підготовці даних для оновлення ваг: {str(e)}")
            return [], []

    def _update_signal_weights(self, category, correct_predictions, wrong_predictions):
        """
        Оновлення ваг сигналів на основі правильних та неправильних передбачень

        :param category: Категорія токена
        :param correct_predictions: Список правильних передбачень
        :param wrong_predictions: Список неправильних передбачень
        """
        # Якщо є дані для оновлення
        if correct_predictions and wrong_predictions:
            print(f"Оновлення ваг сигналів для категорії {category}...")
            try:
                # Оновлюємо ваги через SignalWeightsManager
                updated_weights = self.adaptive_detector.weights_manager.update_weights(
                    category, correct_predictions, wrong_predictions
                )

                # Явно зберігаємо ваги
                self.adaptive_detector.weights_manager.save_weights()
                print(f"Ваги сигналів успішно оновлено та збережено для категорії {category}")
            except Exception as e:
                print(f"Помилка при оновленні ваг сигналів: {str(e)}")
        else:
            print("Недостатньо даних для оновлення ваг сигналів")
