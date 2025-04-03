import os
import json

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import crypto_detector.config.settings as settings

from crypto_detector.models.ml_trainer import MLTrainer
from crypto_detector.models.signal_weights import SignalWeightsManager
from crypto_detector.data.data_storage import DataStorage


class AdaptiveCryptoDetector:
    """
    Адаптивний детектор криптовалютної активності з механізмом самонавчання.
    Розширює базовий детектор, додаючи можливість адаптації ваг сигналів та
    використання ML моделей для покращення точності виявлення.
    """

    def __init__(self, base_detector, model_dir="data/models"):
        """
        Ініціалізація адаптивного детектора

        :param base_detector: Базовий екземпляр CryptoActivityDetector
        :param model_dir: Директорія для збереження навчених моделей
        """
        self.base_detector = base_detector
        self.model_dir = model_dir

        # Створення директорії для моделей, якщо не існує
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Ініціалізація компонентів для зберігання, навчання і управління вагами
        self.storage = DataStorage()
        self.ml_trainer = MLTrainer(storage=self.storage)
        self.weights_manager = SignalWeightsManager(storage=self.storage)

        # Отримання ваг сигналів з weights_manager
        self.default_weights = self.weights_manager.default_weights
        self.token_weights = self.weights_manager.token_weights

        # Словник моделей для різних типів токенів
        self.ml_models = {}

        # Історія результатів для навчання
        self.training_history = {}

        # Мапа категорій токенів
        self.token_categories = settings.TOKEN_CATEGORIES

        # Завантаження збережених моделей, якщо вони існують
        self._load_saved_models()

        # Навчальні гіперпараметри
        self.learning_rate = settings.LEARNING_RATE
        self.training_threshold = settings.TRAINING_THRESHOLD
        self.retraining_interval = settings.RETRAINING_INTERVAL

    def _get_token_category(self, symbol):
        """
        Визначення категорії токена

        :param symbol: Символ криптовалюти (наприклад, 'BTC/USDT')
        :return: Категорія токена
        """
        base_currency = symbol.split('/')[0]

        for category, tokens in self.token_categories.items():
            if base_currency in tokens:
                return category

        return 'other'

    def _load_saved_models(self):
        """Завантаження збережених ML моделей"""
        # Створюємо список усіх категорій, додавши 'other'
        all_categories = list(self.token_categories.keys()) + ['other']

        # Перевіряємо директорію моделей
        if not os.path.exists(self.model_dir):
            print(f"Директорія моделей {self.model_dir} не існує. Створюємо нову.")
            os.makedirs(self.model_dir)
            return

        # Завантажуємо моделі
        for category in all_categories:
            model_path = os.path.join(self.model_dir, f'{category}_model.pkl')
            if os.path.exists(model_path):
                try:
                    self.ml_models[category] = joblib.load(model_path)
                    print(f"Завантажено ML модель для категорії {category}")
                except Exception as e:
                    print(f"Помилка завантаження моделі для {category}: {str(e)}")

    def _save_ml_model(self, category):
        """
        Збереження ML моделі для конкретної категорії

        :param category: Категорія токена
        """
        if category in self.ml_models:
            model_path = os.path.join(self.model_dir, f'{category}_model.pkl')
            try:
                joblib.dump(self.ml_models[category], model_path)
                print(f"Збережено ML модель для категорії {category}")
            except Exception as e:
                print(f"Помилка збереження моделі для {category}: {str(e)}")

    def get_weights_for_token(self, symbol):
        """
        Отримання оптимізованих ваг для конкретного токена

        :param symbol: Символ криптовалюти
        :return: Словник з вагами сигналів
        """
        return self.weights_manager.get_weights_for_token(symbol)

    async def analyze_token(self, symbol, all_symbols=None):
        """
        Аналіз токена з використанням адаптивних ваг та ML моделей

        :param symbol: Символ криптовалюти
        :param all_symbols: Список всіх символів для аналізу кореляції
        :return: Результати аналізу
        """
        # Отримуємо базовий результат від CryptoActivityDetector
        base_result = await self.base_detector.analyze_token(symbol, all_symbols)

        # Категорія токена
        category = self._get_token_category(symbol)

        # Отримуємо оптимізовані ваги для категорії
        weights = self.get_weights_for_token(symbol)

        # Переобчислення оцінки ймовірності з оптимізованими вагами
        confidence = 0.0
        for signal in base_result['signals']:
            signal_name = signal['name']

            # Оновлюємо вагу сигналу з оптимізованих ваг
            if signal_name in weights:
                signal['weight'] = weights[signal_name]

            # Переобчислюємо внесок сигналу відповідно до його ваги
            if signal_name == 'Аномальний обсяг торгів':
                percent_change = base_result['raw_data']['volume'].get('percent_change', 0)
                confidence += weights[signal_name] * min(percent_change / 80, 1.0) if percent_change > 0 else 0

            elif signal_name == 'Активна цінова динаміка':
                price_change = base_result['raw_data']['price'].get('price_change_1h', 0)
                confidence += weights[signal_name] * min(abs(price_change) / 8, 1.0)

            elif signal_name == 'Дисбаланс книги ордерів':
                ratio = base_result['raw_data']['order_book'].get('buy_sell_ratio', 1.0)
                confidence += weights[signal_name] * (min(ratio / 1.7, 1.0) if ratio > 1 else min(1 / ratio / 1.7, 1.0))

            elif signal_name == 'Підвищена активність у соціальних мережах':
                social_change = base_result['raw_data']['social'].get('percent_change', 0)
                confidence += weights[signal_name] * min(social_change / 80, 1.0)

            elif signal_name == 'Нові лістинги на біржах':
                confidence += weights[signal_name]

            elif signal_name == 'Підозрілий часовий патерн':
                time_risk = base_result['raw_data']['time_pattern'].get('time_risk_score', 0)
                confidence += weights[signal_name] * time_risk

            elif signal_name == 'Корельована активність з іншими монетами':
                confidence += weights[signal_name]

            elif signal_name == 'Прискорення зростання об\'єму':
                volume_accel = base_result['raw_data']['volume'].get('volume_acceleration', 0)
                confidence += weights[signal_name] * min(volume_accel / 0.2, 1.0)

            elif signal_name == 'Вертикальний стрибок ціни':
                jump_percent = base_result['raw_data']['price'].get('jump_percent', 0)
                confidence += weights[signal_name] * min(jump_percent / 50, 1.0)

            elif signal_name == 'V-подібний патерн ціни':
                confidence += weights[signal_name]

            elif signal_name == 'Велика зелена свічка з довгим тілом':
                candle_body_percent = base_result['raw_data']['price'].get('candle_body_percent', 0)
                confidence += weights[signal_name] * min(candle_body_percent / 15, 1.0)

            elif signal_name == 'Сильна кореляційна група':
                correlated_coins = len(base_result['raw_data']['correlation'].get('correlated_coins', []))
                confidence += weights[signal_name] * min(correlated_coins / 5, 1.0)

            elif signal_name == 'Синхронізований pump у групі монет':
                price_change = base_result['raw_data']['correlation'].get('price_change_1h', 0)
                confidence += weights[signal_name] * min(price_change / 15, 1.0)

            elif signal_name in ['Виявлено стіну ордерів купівлі', 'Виявлено стіну ордерів продажу']:
                confidence += weights[signal_name]

            elif signal_name == 'Висока концентрація об\'єму':
                volume_concentration = base_result['raw_data']['order_book'].get('volume_concentration', 1.0)
                confidence += weights[signal_name] * min(volume_concentration / 4, 1.0)

        features = []
        ml_probability = None

        if category in self.ml_models:
            try:
                features = self._extract_features_from_result(base_result)

                print(f"Extracted {len(features)} features for ML prediction")

                features_array = np.array(features).reshape(1, -1)

                expected_features = 33

                if features_array.shape[1] != expected_features:
                    print(
                        f"WARNING: Feature count mismatch before prediction - got {features_array.shape[1]}, expected {expected_features}")

                    if features_array.shape[1] > expected_features:
                        features_array = features_array[:, :expected_features]
                    else:
                        features_array = np.pad(features_array,
                                                ((0, 0), (0, expected_features - features_array.shape[1])), 'constant')

                ml_probability = self.ml_models[category].predict_proba(features_array)[0][1]

                print(f"ML prediction successful, probability: {ml_probability:.4f}")

            except Exception as e:
                print(f"Помилка при використанні ML моделі: {str(e)}")
                import traceback
                traceback.print_exc()

        # Комбінуємо оцінки (евристичну та ML, якщо є)
        final_probability = confidence
        if ml_probability is not None:
            # Даємо вагу 0.7 ML моделі і 0.3 евристичній оцінці
            final_probability = 0.3 * confidence + 0.7 * ml_probability

        # Оновлюємо результат
        result = base_result.copy()
        result['probability_score'] = min(final_probability, 1.0)
        result['ml_probability'] = ml_probability
        result['weighted_signals'] = result['signals']  # Сигнали вже з оновленими вагами

        # Додаємо результат до історії для подальшого навчання
        if category not in self.training_history:
            self.training_history[category] = []

        self.training_history[category].append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'features': features,
            'probability': result['probability_score'],
            'signals': {s['name']: s['weight'] for s in result['signals']},
            'actual_event': None  # Буде заповнено пізніше через feedback
        })

        return result

    def _extract_features_from_result(self, result):
        """
        Витягування ознак з результату для ML моделі з урахуванням нових метрик pump-and-dump

        :param result: Результат аналізу
        :return: Список ознак
        """
        features = []

        # Ознаки об'єму
        volume_data = result['raw_data']['volume']
        features.extend([
            volume_data.get('percent_change', 0) / 100,
            volume_data.get('z_score', 0),
            volume_data.get('anomaly_count', 0) / 5,
            volume_data.get('volume_acceleration', 0)
        ])

        # Ознаки ціни - оновлені з новими метриками
        price_data = result['raw_data']['price']
        features.extend([
            price_data.get('price_change_1h', 0) / 100,
            price_data.get('price_change_24h', 0) / 100,  # Важлива метрика для pump-and-dump
            price_data.get('volatility_ratio', 0),
            price_data.get('large_candles', 0) / 5,
            price_data.get('consecutive_up', 0) / 5,
            price_data.get('price_acceleration', 0)
        ])

        # Додаємо нові ознаки для виявлення pump-and-dump
        distance_from_high = price_data.get('distance_from_high', 0) / 100
        dump_phase = 1 if price_data.get('dump_phase', False) else 0
        significant_pump = 1 if price_data.get('significant_pump', False) else 0
        price_above_ema = 1 if price_data.get('price_above_ema', False) else 0

        # Додаємо нові метрики до списку ознак
        features.extend([
            distance_from_high,
            dump_phase,
            significant_pump,
            price_above_ema
        ])

        # Ознаки книги ордерів
        orderbook_data = result['raw_data']['order_book']
        features.extend([
            orderbook_data.get('buy_sell_ratio', 1.0),
            orderbook_data.get('top_concentration', 0),
            1 if orderbook_data.get('has_buy_wall', False) else 0,
            1 if orderbook_data.get('has_sell_wall', False) else 0,
            orderbook_data.get('volume_concentration', 1.0)
        ])

        # Ознаки соціальних даних
        social_data = result['raw_data']['social']
        features.extend([
            social_data.get('percent_change', 0) / 100,
            social_data.get('growth_acceleration', 0)
        ])

        # Ознаки часових патернів
        time_data = result['raw_data']['time_pattern']
        features.extend([
            time_data.get('time_risk_score', 0),
            1 if time_data.get('is_high_risk_hour', False) else 0,
            1 if time_data.get('is_weekend', False) else 0
        ])

        # Ознаки кореляції ринків
        correlation_data = result['raw_data']['correlation']
        features.extend([
            1 if correlation_data.get('correlation_signal', False) else 0,
            len(correlation_data.get('correlated_coins', [])),
            1 if correlation_data.get('correlation_type') == 'pump_group' else 0,
            correlation_data.get('correlation_strength', 0.0)
        ])

        # Вертикальний стрибок ціни
        vertical_jump = 1 if price_data.get('vertical_price_jump', False) else 0
        jump_percent = price_data.get('jump_percent', 0) / 100  # Нормалізація

        # V-подібний патерн
        v_pattern = 1 if price_data.get('v_pattern_detected', False) else 0

        # Велика зелена свічка
        large_green_candle = 1 if price_data.get('large_green_candle', False) else 0
        candle_body_percent = price_data.get('candle_body_percent', 0) / 100  # Нормалізація

        # Додавання нових ознак до загального списку
        features.extend([
            vertical_jump,
            jump_percent,
            v_pattern,
            large_green_candle,
            candle_body_percent,
        ])

        expected_features = 33

        if len(features) != expected_features:
            print(f"WARNING: Feature count mismatch - got {len(features)}, expected {expected_features}")

            if len(features) > expected_features:
                features = features[:expected_features]
            else:
                features.extend([0] * (expected_features - len(features)))

        return features

    def provide_feedback(self, symbol, timestamp, is_event):
        """
        Надання зворотного зв'язку про фактичний результат події

        :param symbol: Символ криптовалюти
        :param timestamp: Часова мітка аналізу
        :param is_event: Чи була подія pump-and-dump (True/False)
        """
        category = self._get_token_category(symbol)

        if category in self.training_history:
            for entry in self.training_history[category]:
                if entry['symbol'] == symbol and (
                        datetime.fromisoformat(timestamp) - entry['timestamp']).total_seconds() < 300:
                    entry['actual_event'] = is_event
                    print(f"Додано зворотний зв'язок для {symbol} на {timestamp}: {is_event}")
                    break

            self._update_weights_using_manager(category)

            labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]

            if len(labeled_entries) >= self.retraining_interval:
                self._train_ml_model_with_trainer(category)

    def _update_weights_using_manager(self, category):
        """
        Оновлення ваг за допомогою SignalWeightsManager

        :param category: Категорія токена
        """
        # Отримуємо записи з відомими результатами
        labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]

        if len(labeled_entries) < self.training_threshold:
            return

        # Розділяємо на успішні та неуспішні передбачення
        correct_predictions = []
        wrong_predictions = []

        for entry in labeled_entries:
            predicted_event = entry['probability'] > 0.5
            if predicted_event == entry['actual_event']:
                correct_predictions.append(entry)
            else:
                wrong_predictions.append(entry)

        # Якщо немає помилкових передбачень, нічого оновлювати
        if not wrong_predictions:
            return

        # Використовуємо SignalWeightsManager для оновлення ваг
        self.weights_manager.update_weights(category, correct_predictions, wrong_predictions)

        # Оновлюємо локальну копію ваг
        self.token_weights = self.weights_manager.token_weights

        print(
            f"Оновлено ваги для категорії {category}. Правильних: {len(correct_predictions)}, неправильних: {len(wrong_predictions)}")

    def _train_ml_model_with_trainer(self, category):
        """
        Навчання ML моделі з використанням MLTrainer

        :param category: Категорія токена
        """
        # Перетворюємо дані історії в придатний для навчання формат
        labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]

        if len(labeled_entries) < self.retraining_interval:
            return

        print(f"Підготовка даних для навчання ML моделі категорії {category}...")

        features_list = [e['features'] for e in labeled_entries]
        labels = [1 if e['actual_event'] else 0 for e in labeled_entries]

        expected_features = 33
        standardized_features = []

        for features in features_list:
            if len(features) != expected_features:
                print(f"Adjusting feature count: {len(features)} -> {expected_features}")

                if len(features) > expected_features:
                    features = features[:expected_features]
                else:
                    features = features + [0] * (expected_features - len(features))

            standardized_features.append(features)

        temp_dir = os.path.join(self.model_dir, "temp")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        training_data = pd.DataFrame(standardized_features)
        training_data['is_event'] = labels

        feature_names = [
            # Ознаки об'єму (4)
            'volume_percent_change', 'volume_z_score', 'volume_anomaly_count', 'volume_acceleration',
            # Ознаки ціни (6)
            'price_change_1h', 'price_change_24h', 'volatility_ratio', 'large_candles',
            'consecutive_up', 'price_acceleration',
            # Додаткові ознаки pump-and-dump (4)
            'distance_from_high', 'dump_phase', 'significant_pump', 'price_above_ema',
            # Ознаки книги ордерів (4)
            'buy_sell_ratio', 'top_concentration', 'has_buy_wall', 'has_sell_wall', 'volume_concentration',
            # Ознаки соціальних даних (2)
            'social_percent_change', 'social_growth_acceleration',
            # Ознаки часових патернів (3)
            'time_risk_score', 'is_high_risk_hour', 'is_weekend',
            # Ознака кореляції (1)
            'correlation_signal', 'correlated_coins_count', 'correlation_type_pump_group', 'correlation_strength',
            # Нові ознаки патернів (5)
            'vertical_jump', 'jump_percent', 'v_pattern', 'large_green_candle', 'candle_body_percent'
        ]

        if len(feature_names) != training_data.shape[1] - 1:
            print(
                f"WARNING: Feature names count ({len(feature_names)}) doesn't match data columns ({training_data.shape[1] - 1})")

            if len(feature_names) < training_data.shape[1] - 1:
                for i in range(len(feature_names), training_data.shape[1] - 1):
                    feature_names.append(f'feature_{i}')
            else:
                feature_names = feature_names[:(training_data.shape[1] - 1)]

        # Перейменовуємо колонки
        column_mapping = {i: name for i, name in enumerate(feature_names)}
        column_mapping[training_data.shape[1] - 1] = 'is_event'
        training_data = training_data.rename(columns=column_mapping)

        # Зберігаємо дані у тимчасовий файл
        training_file = os.path.join(temp_dir, f"{category}_training_data.csv")
        training_data.to_csv(training_file, index=False)

        print(f"Навчання моделі для категорії {category} з {len(labeled_entries)} зразків...")

        try:
            X, y = self.ml_trainer.prepare_features(training_file)

            if X.shape[1] != expected_features:
                print(
                    f"WARNING: Features shape after prepare_features: {X.shape}, expected ({len(labeled_entries)}, {expected_features})")

                X = self.ml_trainer.prepare_model_features(X)
                print(f"Features shape after adjustment: {X.shape}")

            print(f"Оптимізація гіперпараметрів для {category}...")
            best_params, _ = self.ml_trainer.optimize_hyperparameters(X, y, model_type='gradient_boosting',
                                                                      category=category)

            print(f"Тренування моделі для {category} з оптимальними параметрами: {best_params}")

            model, info = self.ml_trainer.train_model(X, y, model_type='gradient_boosting', category=category,
                                                      **best_params)

            model_params_file = os.path.join(self.model_dir, f"{category}_best_params.json")

            with open(model_params_file, 'w') as f:
                json.dump(best_params, f, indent=4)

            self.ml_models[category] = model
            self._save_ml_model(category)

            charts_dir = os.path.join(self.model_dir, "charts")

            if not os.path.exists(charts_dir):
                os.makedirs(charts_dir)

            self.ml_trainer.visualize_feature_importance(
                info['feature_importance'],
                title=f"Feature Importance for {category}",
                top_n=20
            )

            cv_report, _ = self.ml_trainer.cross_validate_and_report(X, y, model_type='gradient_boosting',
                                                                     category=category, **best_params)

            report_file = os.path.join(charts_dir, f"{category}_ml_report.json")

            with open(report_file, 'w') as f:
                json.dump(cv_report, f, indent=4, default=str)

            print(f"Навчання ML моделі для категорії {category} завершено успішно.")
            print(
                f"Метрики: Precision={info['metrics']['precision']:.4f}, Recall={info['metrics']['recall']:.4f}, F1={info['metrics']['f1_score']:.4f}")

        except Exception as e:
            print(f"Помилка при навчанні ML моделі для категорії {category}: {str(e)}")
            import traceback
            traceback.print_exc()

        self.training_history[category] = self.training_history[category][-50:]

    def get_optimized_threshold(self, symbol):
        """
        Отримання оптимізованого порогу для класифікації

        :param symbol: Символ криптовалюти
        :return: Оптимізований поріг
        """
        category = self._get_token_category(symbol)

        # Спроба отримати оптимізований поріг з історичних даних
        try:
            if category in self.training_history:
                labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]
                if len(labeled_entries) >= 30:
                    return self.weights_manager.find_optimal_threshold(labeled_entries, category)[0]
        except Exception as e:
            print(f"Помилка при пошуку оптимального порогу: {str(e)}")

        # Повернення порогу з налаштувань, якщо немає оптимізованого порогу
        from crypto_detector.config.settings import TOKEN_THRESHOLDS
        return TOKEN_THRESHOLDS.get(category, TOKEN_THRESHOLDS['other'])

    async def analyze_historical_point(self, symbol, data_window, timestamp):
        """
        Аналіз історичної точки даних з використанням адаптивних ваг

        :param symbol: Символ криптовалюти
        :param data_window: Вікно історичних даних
        :param timestamp: Часова мітка
        :return: Результат аналізу
        """
        # Аналіз об'єму
        volume_analysis = await self.base_detector.volume_analyzer.analyze_historical_volume(data_window)

        # Аналіз цінової динаміки
        price_analysis = await self.base_detector.price_analyzer.analyze_historical_price(data_window)

        # Виявлення нових патернів
        vertical_jump, jump_percent = self.base_detector.price_analyzer.detect_vertical_price_jump(data_window)
        v_pattern_detected = self.base_detector.price_analyzer.detect_v_pattern(data_window)
        large_green_candle, candle_body_percent = self.base_detector.price_analyzer.detect_large_candles(data_window)

        # Додаємо результати до price_analysis
        price_analysis['vertical_price_jump'] = vertical_jump
        price_analysis['jump_percent'] = jump_percent
        price_analysis['v_pattern_detected'] = v_pattern_detected
        price_analysis['large_green_candle'] = large_green_candle
        price_analysis['candle_body_percent'] = candle_body_percent

        # Додаткові розрахунки для прискорення об'єму
        if not data_window.empty and len(data_window) >= 6:
            data_window['volume_change'] = data_window['volume'].pct_change()
            volume_acceleration = data_window['volume_change'].diff()[-5:].mean()
            volume_analysis['volume_acceleration'] = volume_acceleration
        else:
            volume_analysis['volume_acceleration'] = 0

        # Імітація аналізу книги ордерів
        order_book_analysis = await self.base_detector.orderbook_analyzer._analyze_historical_orderbook(timestamp, data_window)

        # Імітація соціальних даних
        social_data = {
            'social_signal': False,
            'mentions': 0,
            'average_mentions': 0,
            'percent_change': 0
        }

        # Імітація даних кореляції
        correlation_data = await self._analyze_historical_correlation(
            self.base_detector, symbol, timestamp, data_window
        )

        # Імітація часових патернів
        time_of_day = pd.Timestamp(timestamp).hour
        is_night = 0 <= time_of_day <= 4 or 20 <= time_of_day <= 23
        is_weekend = pd.Timestamp(timestamp).dayofweek >= 5  # 5 = Saturday, 6 = Sunday

        time_pattern_data = {
            'time_pattern_signal': is_night or is_weekend,
            'is_high_risk_hour': is_night,
            'is_weekend': is_weekend,
            'time_risk_score': 0.7 if is_night else (0.3 if is_weekend else 0.0)
        }

        # Сигнали
        signals = []
        confidence = 0.0

        # Отримуємо оптимізовані ваги через SignalWeightsManager
        weights = self.get_weights_for_token(symbol)

        # Сигнал 1: Аномальний обсяг торгів
        if volume_analysis['unusual_volume']:
            signals.append({
                'name': 'Аномальний обсяг торгів',
                'description': f"Поточний обсяг перевищує середній на {volume_analysis['percent_change']:.2f}%",
                'weight': weights.get('Аномальний обсяг торгів', 0.35)
            })
            confidence += weights.get('Аномальний обсяг торгів', 0.35) * min(volume_analysis['percent_change'] / 80,
                                                                             1.0) if volume_analysis[
                                                                                         'percent_change'] > 0 else 0

        # Сигнал 2: Цінова динаміка
        if price_analysis['price_action_signal']:
            signals.append({
                'name': 'Активна цінова динаміка',
                'description': f"Зміна ціни за останній період: {price_analysis['recent_price_change']:.2f}%",
                'weight': weights.get('Активна цінова динаміка', 0.25)
            })
            confidence += weights.get('Активна цінова динаміка', 0.25) * min(
                abs(price_analysis['recent_price_change']) / 8, 1.0)

        # Сигнал 3: Дисбаланс книги ордерів
        if order_book_analysis['order_book_signal']:
            signals.append({
                'name': 'Дисбаланс книги ордерів',
                'description': f"Співвідношення ордерів купівлі/продажу: {order_book_analysis['buy_sell_ratio']:.2f}",
                'weight': 0.2
            })
            ratio = order_book_analysis['buy_sell_ratio']

            if ratio == 0:
                confidence += 0.2
            elif ratio > 1:
                confidence += 0.2 * min(ratio / 1.7, 1.0)
            else:
                confidence += 0.2 * min(1 / ratio / 1.7, 1.0)

        # Сигнал 5: Часовий патерн
        if time_pattern_data['time_pattern_signal']:
            signals.append({
                'name': 'Підозрілий часовий патерн',
                'description': f"Поточний час відповідає високоризиковому періоду для pump-and-dump схем",
                'weight': weights.get('Підозрілий часовий патерн', 0.15)
            })
            confidence += weights.get('Підозрілий часовий патерн', 0.15) * time_pattern_data['time_risk_score']

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
        if volume_analysis.get('volume_acceleration', 0) > 0.05:
            signals.append({
                'name': 'Прискорення зростання об\'єму',
                'description': f"Швидкість зростання об\'єму збільшується: {volume_analysis['volume_acceleration']:.2f}",
                'weight': weights.get('Прискорення зростання об\'єму', 0.25)
            })
            confidence += weights.get('Прискорення зростання об\'єму', 0.25) * min(
                volume_analysis['volume_acceleration'] / 0.15, 1.0)

        # Сигнал 8: Значна зміна ціни за 24 години
        if price_analysis.get('price_change_24h', 0) > 50:
            signals.append({
                'name': 'Значна зміна ціни за 24 години',
                'description': f"Ціна зросла на {price_analysis['price_change_24h']:.2f}% за останні 24 години",
                'weight': 0.40
            })
            confidence += 0.40 * min(price_analysis['price_change_24h'] / 100, 1.0)

        # Сигнал 9: Виявлено dump фазу після pump
        if price_analysis.get('dump_phase', False):
            signals.append({
                'name': 'Dump фаза після pump',
                'description': f"Ціна знизилась на {abs(price_analysis['distance_from_high']):.2f}% від нещодавнього піку",
                'weight': 0.35
            })
            confidence += 0.35 * min(abs(price_analysis['distance_from_high']) / 30, 1.0)

        # Новий сигнал 10: Вертикальний стрибок ціни
        if price_analysis.get('vertical_price_jump', False):
            signals.append({
                'name': 'Вертикальний стрибок ціни',
                'description': f"Виявлено вертикальне зростання ціни на {price_analysis.get('jump_percent', 0):.2f}% за короткий період",
                'weight': weights.get('Вертикальний стрибок ціни', 0.40)
            })
            confidence += weights.get('Вертикальний стрибок ціни', 0.40) * min(
                price_analysis.get('jump_percent', 0) / 50, 1.0)

        # Новий сигнал 11: Паттерн V-подібного руху ціни
        if price_analysis.get('v_pattern_detected', False):
            signals.append({
                'name': 'V-подібний патерн ціни',
                'description': f"Виявлено швидке зростання і падіння ціни без консолідації",
                'weight': weights.get('V-подібний патерн ціни', 0.35)
            })
            confidence += weights.get('V-подібний патерн ціни', 0.35)

        # Новий сигнал 12: Велика зелена свічка з довгим тілом
        if price_analysis.get('large_green_candle', False):
            signals.append({
                'name': 'Велика зелена свічка з довгим тілом',
                'description': f"Тіло свічки складає {price_analysis.get('candle_body_percent', 0):.2f}% від ціни",
                'weight': weights.get('Велика зелена свічка з довгим тілом', 0.30)
            })
            confidence += weights.get('Велика зелена свічка з довгим тілом', 0.30) * min(
                price_analysis.get('candle_body_percent', 0) / 15, 1.0)

        # Сигнал 13: Сильна кореляційна група
        if correlation_data.get('correlated_coins') and len(correlation_data['correlated_coins']) >= 3:
            signals.append({
                'name': 'Сильна кореляційна група',
                'description': f"Монета входить до групи з {len(correlation_data['correlated_coins']) + 1} корельованих активів",
                'weight': 0.25
            })
            confidence += 0.25 * min(len(correlation_data['correlated_coins']) / 5, 1.0)

        # Сигнал 14: Синхронізований pump у кореляційній групі
        if (correlation_data.get('correlation_type') == 'pump_group' and
                correlation_data.get('price_change_1h', 0) > 5.0):
            signals.append({
                'name': 'Синхронізований pump у групі монет',
                'description': f"Виявлено синхронний pump з іншими монетами, зміна ціни: {correlation_data.get('price_change_1h', 0):.2f}%",
                'weight': 0.35
            })
            confidence += 0.35 * min(correlation_data.get('price_change_1h', 0) / 15, 1.0)

        # Сигнал 15: Наявність стіни ордерів
        if order_book_analysis.get('has_buy_wall', False) or order_book_analysis.get('has_sell_wall', False):
            wall_type = "купівлі" if order_book_analysis.get('has_buy_wall', False) else "продажу"
            signals.append({
                'name': f'Виявлено стіну ордерів {wall_type}',
                'description': f"Значна концентрація ліквідності виявлена у книзі ордерів",
                'weight': 0.25
            })
            confidence += 0.25

        # Формування результату
        result = {
            'symbol': symbol,
            'timestamp': pd.Timestamp(timestamp).isoformat(),
            'probability_score': min(confidence, 1.0),
            'signals': signals,
            'raw_data': {
                'volume': volume_analysis,
                'price': price_analysis,
                'time_pattern': time_pattern_data,
                'order_book': order_book_analysis,
                'social': social_data,
                'correlation': correlation_data
            }
        }

        return result

    def evaluate_model_performance(self, category, test_data=None):
        """
        Оцінка ефективності моделі для заданої категорії

        :param category: Категорія токена
        :param test_data: Тестові дані (якщо None, використовується remaining_history)
        :return: Метрики ефективності
        """
        if category not in self.ml_models:
            print(f"Немає моделі для категорії {category}")
            return None

        if test_data is None:
            labeled_entries = [e for e in self.training_history.get(category, []) if e['actual_event'] is not None]
            if len(labeled_entries) < 10:
                print(f"Недостатньо даних для оцінки моделі категорії {category}")
                return None

            split_idx = int(len(labeled_entries) * 0.8)
            test_entries = labeled_entries[split_idx:]

            features = [e['features'] for e in test_entries]
            true_labels = [1 if e['actual_event'] else 0 for e in test_entries]
        else:
            try:
                X, y = self.ml_trainer.prepare_features(test_data)
                features = X
                true_labels = y
            except Exception as e:
                print(f"Помилка при підготовці тестових даних: {str(e)}")
                return None

        model = self.ml_models[category]

        try:
            probabilities = model.predict_proba(features)[:, 1]

            optimal_threshold = self.weights_manager.find_optimal_threshold(
                self.training_history.get(category, []), category)[0]

            if not optimal_threshold:
                optimal_threshold = 0.5

            predictions = [1 if p > optimal_threshold else 0 for p in probabilities]

            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)

            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'optimal_threshold': optimal_threshold,
                'sample_size': len(true_labels)
            }

            print(f"Оцінка ефективності моделі для категорії {category}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Оптимальний поріг: {optimal_threshold:.4f}")
            print(f"  Розмір вибірки: {len(true_labels)}")

            return metrics

        except Exception as e:
            print(f"Помилка при оцінці ефективності моделі: {str(e)}")
            return None

    def visualize_weights_and_signal_importance(self, category):
        """
        Візуалізація ваг сигналів та їх важливості для заданої категорії

        :param category: Категорія токена
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if category not in self.token_weights:
            print(f"Немає оптимізованих ваг для категорії {category}")
            return

        # Отримання ваг для категорії
        optimized_weights = self.token_weights[category]
        default_weights = self.default_weights

        # Створення DataFrame для візуалізації
        weights_df = pd.DataFrame({
            'Signal': list(optimized_weights.keys()),
            'Default Weight': [default_weights.get(s, 0) for s in optimized_weights.keys()],
            'Optimized Weight': list(optimized_weights.values())
        })

        # Сортування за зміною ваги
        weights_df['Weight Change'] = weights_df['Optimized Weight'] - weights_df['Default Weight']
        weights_df = weights_df.sort_values('Weight Change', ascending=False)

        # Створення директорії для графіків, якщо вона не існує
        charts_dir = os.path.join(self.model_dir, "charts")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        # Візуалізація порівняння ваг
        plt.figure(figsize=(12, 8))

        sns.barplot(x='Signal', y='value', hue='variable',
                    data=pd.melt(weights_df, id_vars=['Signal'],
                                 value_vars=['Default Weight', 'Optimized Weight']))

        plt.title(f'Порівняння ваг сигналів для категорії {category}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Збереження графіка
        plt.savefig(os.path.join(charts_dir, f"{category}_weights_comparison.png"))
        plt.close()

        # Якщо є ML модель, візуалізуємо також важливість ознак
        if category in self.ml_models and hasattr(self.ml_models[category], 'feature_importances_'):
            # Отримання важливості ознак
            feature_names = [
                'volume_percent_change', 'volume_z_score', 'volume_anomaly_count', 'volume_acceleration',
                'price_change_1h', 'price_change_24h', 'volatility_ratio', 'large_candles', 'consecutive_up',
                'price_acceleration',
                'buy_sell_ratio', 'top_concentration', 'spread', 'has_buy_wall', 'has_sell_wall',
                'social_percent_change', 'social_growth_acceleration',
                'time_risk_score', 'is_high_risk_hour', 'is_weekend',
                'correlation_signal'
            ]

            feature_importances = self.ml_models[category].feature_importances_

            # Створення DataFrame для візуалізації
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(feature_importances)],
                'Importance': feature_importances
            })

            # Сортування за важливістю
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # Візуалізація важливості ознак
            plt.figure(figsize=(12, 8))

            sns.barplot(x='Importance', y='Feature', data=importance_df)

            plt.title(f'Важливість ознак для категорії {category}')
            plt.tight_layout()

            # Збереження графіка
            plt.savefig(os.path.join(charts_dir, f"{category}_feature_importance.png"))
            plt.close()

            print(f"Графіки збережено у директорії {charts_dir}")

    def export_trained_model(self, category, export_dir=None):
        """
        Експорт натренованої моделі для використання в інших системах

        :param category: Категорія токена
        :param export_dir: Директорія для експорту (якщо None, використовується model_dir/export)
        :return: Шлях до експортованого файлу
        """
        if category not in self.ml_models:
            print(f"Немає моделі для категорії {category}")
            return None

        # Визначення директорії для експорту
        if export_dir is None:
            export_dir = os.path.join(self.model_dir, "export")

        # Створення директорії, якщо вона не існує
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        try:
            # Експорт моделі
            model_path = os.path.join(export_dir, f"{category}_model.pkl")
            joblib.dump(self.ml_models[category], model_path)

            # Експорт ваг сигналів
            weights_path = os.path.join(export_dir, f"{category}_weights.json")
            with open(weights_path, 'w') as f:
                json.dump(self.token_weights.get(category, self.default_weights), f, indent=4)

            # Експорт метаданих
            metadata = {
                'category': category,
                'date_exported': datetime.now().isoformat(),
                'model_type': type(self.ml_models[category]).__name__,
                'feature_names': [
                    'volume_percent_change', 'volume_z_score', 'volume_anomaly_count', 'volume_acceleration',
                    'price_change_1h', 'price_change_24h', 'volatility_ratio', 'large_candles', 'consecutive_up',
                    'price_acceleration',
                    'buy_sell_ratio', 'top_concentration', 'spread', 'has_buy_wall', 'has_sell_wall',
                    'social_percent_change', 'social_growth_acceleration',
                    'time_risk_score', 'is_high_risk_hour', 'is_weekend',
                    'correlation_signal'
                ],
                'threshold': self.get_optimized_threshold(
                    f"{list(self.token_categories.get(category, ['BTC']))[0]}/USDT")
            }

            metadata_path = os.path.join(export_dir, f"{category}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f"Модель для категорії {category} успішно експортовано у {export_dir}")

            return {
                'model': model_path,
                'weights': weights_path,
                'metadata': metadata_path
            }

        except Exception as e:
            print(f"Помилка при експорті моделі: {str(e)}")
            return None

    def import_trained_model(self, category, import_dir):
        """
        Імпорт натренованої моделі з зовнішнього джерела

        :param category: Категорія токена
        :param import_dir: Директорія з імпортованими файлами
        :return: True в разі успіху, False в разі помилки
        """
        # Перевірка наявності файлів для імпорту
        model_path = os.path.join(import_dir, f"{category}_model.pkl")
        weights_path = os.path.join(import_dir, f"{category}_weights.json")

        if not os.path.exists(model_path) or not os.path.exists(weights_path):
            print(f"Не знайдено необхідні файли для імпорту моделі категорії {category}")
            return False

        try:
            # Імпорт моделі
            model = joblib.load(model_path)
            self.ml_models[category] = model

            # Імпорт ваг сигналів
            with open(weights_path, 'r') as f:
                weights = json.load(f)

            self.token_weights[category] = weights

            # Збереження моделі та ваг у відповідні директорії
            self._save_ml_model(category)
            self.weights_manager.token_weights = self.token_weights
            if hasattr(self.weights_manager, 'save_weights'):
                self.weights_manager.save_weights()

            print(f"Модель для категорії {category} успішно імпортовано")
            return True

        except Exception as e:
            print(f"Помилка при імпорті моделі: {str(e)}")
            return False

    def get_all_models_metrics(self):
        """
        Отримання метрик для всіх натренованих моделей

        :return: Словник з метриками для кожної категорії
        """
        metrics = {}

        for category in self.ml_models.keys():
            try:
                category_metrics = self.evaluate_model_performance(category)
                if category_metrics:
                    metrics[category] = category_metrics
            except Exception as e:
                print(f"Помилка при оцінці моделі категорії {category}: {str(e)}")

        return metrics

    async def _analyze_historical_correlation(self, detector, symbol, timestamp, data_window):
        """
        Аналіз кореляції на історичних даних

        :param detector: Екземпляр CryptoActivityDetector
        :param symbol: Символ криптовалюти
        :param timestamp: Часова мітка аналізу
        :param data_window: Поточне вікно даних
        :return: Результат аналізу кореляції
        """
        # Перевіряємо наявність необхідних атрибутів
        if not hasattr(detector, 'correlation_analyzer') or not hasattr(detector, 'historical_data'):
            return {
                'correlation_signal': False,
                'correlated_coins': [],
                'correlation_type': 'normal'
            }

        try:
            # Отримання часової мітки як об'єкту datetime
            if isinstance(timestamp, str):
                timestamp = pd.Timestamp(timestamp).to_pydatetime()
            elif isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()

            # Визначаємо часові межі для аналізу (3 години до і 1 година після)
            start_time = timestamp - pd.Timedelta(hours=3)
            end_time = timestamp + pd.Timedelta(hours=1)

            # Отримуємо список символів, для яких є історичні дані
            available_symbols = list(detector.historical_data.keys())

            if not available_symbols or symbol not in available_symbols:
                return {
                    'correlation_signal': False,
                    'correlated_coins': [],
                    'correlation_type': 'normal'
                }

            # Отримуємо зміну ціни для поточного символу за останню годину
            if not data_window.empty and len(data_window) > 5:
                last_hour_data = data_window.iloc[-12:]  # Останні 12 5-хвилинних інтервалів = 1 година
                if len(last_hour_data) >= 2:
                    price_change = (last_hour_data['close'].iloc[-1] / last_hour_data['close'].iloc[0] - 1) * 100
                else:
                    price_change = 0
            else:
                price_change = 0

            # Виявляємо різке зростання ціни (pump)
            pump_threshold = 5.0  # Поріг для визначення pump (5% за годину)
            pump_signal = price_change >= pump_threshold

            if not pump_signal:
                return {
                    'correlation_signal': False,
                    'correlated_coins': [],
                    'correlation_type': 'normal',
                    'price_change_1h': price_change
                }

            # Аналіз кореляції з іншими символами
            correlated_coins = []

            for other_symbol in available_symbols:
                if other_symbol == symbol:
                    continue

                # Отримуємо дані для іншого символу
                if other_symbol in detector.historical_data:
                    other_data = detector.historical_data[other_symbol]

                    # Фільтруємо дані за часовим проміжком
                    if isinstance(other_data.index, pd.DatetimeIndex):
                        filtered_data = other_data[(other_data.index >= start_time) & (other_data.index <= end_time)]
                    else:
                        # Якщо індекс не datetime, пропускаємо цей символ
                        continue

                    if filtered_data.empty or len(filtered_data) < 5:
                        continue

                    # Обчислюємо зміну ціни для іншого символу
                    other_last_hour = filtered_data.iloc[-12:]  # Останні 12 5-хвилинних інтервалів
                    if len(other_last_hour) < 2:
                        continue

                    other_price_change = (other_last_hour['close'].iloc[-1] / other_last_hour['close'].iloc[
                        0] - 1) * 100

                    # Визначаємо кореляцію на основі синхронних цінових рухів
                    if other_price_change >= pump_threshold * 0.7:  # Допускаємо невелике відхилення
                        # Розраховуємо кореляцію між двома часовими рядами
                        # Знаходимо спільний часовий проміжок
                        common_data = pd.merge(
                            data_window['close'].pct_change(),
                            filtered_data['close'].pct_change(),
                            left_index=True, right_index=True,
                            suffixes=('_current', '_other'),
                            how='inner'
                        )

                        if len(common_data) > 5:
                            correlation = common_data.corr().iloc[0, 1]

                            # Якщо кореляція висока, додаємо символ до списку корельованих
                            if correlation >= 0.7:  # Високий поріг кореляції
                                correlated_coins.append(other_symbol)

            # Визначаємо тип кореляції
            correlation_type = 'normal'
            if pump_signal and len(correlated_coins) >= 2:
                correlation_type = 'pump_group'
            elif pump_signal:
                correlation_type = 'single_pump'

            correlation_signal = pump_signal and len(correlated_coins) >= 2

            return {
                'correlation_signal': correlation_signal,
                'correlated_coins': correlated_coins,
                'correlation_type': correlation_type,
                'price_change_1h': price_change
            }

        except Exception as e:
            print(f"Помилка аналізу історичної кореляції: {e}")
            return {
                'correlation_signal': False,
                'correlated_coins': [],
                'correlation_type': 'normal'
            }
