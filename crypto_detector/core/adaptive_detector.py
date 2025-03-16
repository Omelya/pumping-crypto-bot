import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import crypto_detector.config.settings as settings

from crypto_detector.core.detector import CryptoActivityDetector


class AdaptiveCryptoDetector:
    """
    Адаптивний детектор криптовалютної активності з механізмом самонавчання.
    Розширює базовий детектор, додаючи можливість адаптації ваг сигналів та
    використання ML моделей для покращення точності виявлення.
    """

    def __init__(self, base_detector, model_dir="models"):
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

        # Дефолтні ваги сигналів
        self.default_weights = settings.DEFAULT_SIGNAL_WEIGHTS

        # Оптимізовані ваги для кожного типу токенів
        self.token_weights = {}

        # Словник моделей для різних типів токенів
        self.ml_models = {}

        # Історія результатів для навчання
        self.training_history = {}

        # Мапа категорій токенів - переміщаємо перед викликом _load_saved_models
        self.token_categories = settings.TOKEN_CATEGORIES

        # Завантаження збережених ваг і моделей, якщо вони існують
        self._load_saved_weights()
        self._load_saved_models()

        # Навчальні гіперпараметри
        self.learning_rate = settings.LEARNING_RATE  # Швидкість, з якою адаптуються ваги
        self.training_threshold = settings.TRAINING_THRESHOLD  # Мінімальна кількість зразків для адаптації ваг
        self.retraining_interval = settings.RETRAINING_INTERVAL  # Інтервал для перенавчання ML моделей

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

    def _load_saved_weights(self):
        """Завантаження збережених ваг з файлу"""
        weights_file = os.path.join(self.model_dir, 'signal_weights.json')

        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    self.token_weights = json.load(f)
                print(f"Завантажено оптимізовані ваги для {len(self.token_weights)} категорій токенів")
            except Exception as e:
                print(f"Помилка завантаження ваг: {str(e)}")

    def _save_weights(self):
        """Збереження оптимізованих ваг у файл"""
        weights_file = os.path.join(self.model_dir, 'signal_weights.json')

        try:
            with open(weights_file, 'w') as f:
                json.dump(self.token_weights, f, indent=4)
        except Exception as e:
            print(f"Помилка збереження ваг: {str(e)}")

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
        category = self._get_token_category(symbol)

        if category in self.token_weights:
            return self.token_weights[category]

        # Якщо ваг для категорії немає, використовуємо дефолтні
        return self.default_weights

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
                percent_change = base_result['raw_data']['volume']['percent_change']
                confidence += weights[signal_name] * min(percent_change / 80, 1.0) if percent_change > 0 else 0

            elif signal_name == 'Активна цінова динаміка':
                price_change = base_result['raw_data']['price']['price_change_1h']
                confidence += weights[signal_name] * min(abs(price_change) / 8, 1.0)

            elif signal_name == 'Дисбаланс книги ордерів':
                ratio = base_result['raw_data']['order_book']['buy_sell_ratio']
                confidence += weights[signal_name] * (min(ratio / 1.7, 1.0) if ratio > 1 else min(1 / ratio / 1.7, 1.0))

            elif signal_name == 'Підвищена активність у соціальних мережах':
                social_change = base_result['raw_data']['social']['percent_change']
                confidence += weights[signal_name] * min(social_change / 80, 1.0)

            elif signal_name == 'Нові лістинги на біржах':
                confidence += weights[signal_name]

            elif signal_name == 'Підозрілий часовий патерн':
                time_risk = base_result['raw_data']['time_pattern']['time_risk_score']
                confidence += weights[signal_name] * time_risk

            elif signal_name == 'Корельована активність з іншими монетами':
                confidence += weights[signal_name]

            elif signal_name == 'Прискорення зростання об\'єму':
                volume_accel = base_result['raw_data']['volume'].get('volume_acceleration', 0)
                confidence += weights[signal_name] * min(volume_accel / 0.2, 1.0)

        # Обчислюємо ML-оцінку, якщо доступна модель для цієї категорії
        ml_probability = None
        if category in self.ml_models:
            try:
                features = self._extract_features_from_result(base_result)
                ml_probability = self.ml_models[category].predict_proba([features])[0][1]
            except Exception as e:
                print(f"Помилка при використанні ML моделі: {str(e)}")

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
            'features': self._extract_features_from_result(base_result),
            'probability': result['probability_score'],
            'signals': {s['name']: s['weight'] for s in result['signals']},
            'actual_event': None  # Буде заповнено пізніше через feedback
        })

        return result

    def _extract_features_from_result(self, result):
        """
        Витягування ознак з результату для ML моделі

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

        # Ознаки ціни
        price_data = result['raw_data']['price']
        features.extend([
            price_data.get('price_change_1h', 0) / 100,
            price_data.get('price_change_24h', 0) / 100,
            price_data.get('volatility_ratio', 0),
            price_data.get('large_candles', 0) / 5,
            price_data.get('consecutive_up', 0) / 5,
            price_data.get('price_acceleration', 0)
        ])

        # Ознаки книги ордерів
        orderbook_data = result['raw_data']['order_book']
        features.extend([
            orderbook_data.get('buy_sell_ratio', 1.0),
            orderbook_data.get('top_concentration', 0),
            orderbook_data.get('spread', 0) / 100,
            1 if orderbook_data.get('has_buy_wall', False) else 0,
            1 if orderbook_data.get('has_sell_wall', False) else 0
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
        features.append(1 if correlation_data.get('correlation_signal', False) else 0)

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
            # Шукаємо відповідний запис в історії
            for entry in self.training_history[category]:
                if entry['symbol'] == symbol and (
                        datetime.fromisoformat(timestamp) - entry['timestamp']).total_seconds() < 300:
                    entry['actual_event'] = is_event
                    print(f"Додано зворотний зв'язок для {symbol} на {timestamp}: {is_event}")
                    break

            # Перевіряємо, чи достатньо даних для адаптації ваг
            self._check_and_update_weights(category)

            # Перевіряємо, чи треба перенавчити ML модель
            labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]
            if len(labeled_entries) >= self.retraining_interval:
                self._train_ml_model(category)

    def _check_and_update_weights(self, category):
        """
        Перевірка та оновлення ваг на основі накопичених даних

        :param category: Категорія токена
        """
        # Отримуємо записи з відомими результатами
        labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]

        if len(labeled_entries) < self.training_threshold:
            return

        # Поточні ваги для категорії
        current_weights = self.token_weights.get(category, self.default_weights)

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

        # Обчислюємо середню активацію сигналів у правильних та помилкових передбаченнях
        signal_importance = {}

        # Для кожного сигналу визначаємо наскільки він був активний у правильних і неправильних передбаченнях
        for signal_name in current_weights.keys():
            correct_activation = sum(1 for e in correct_predictions if signal_name in e['signals'])
            wrong_activation = sum(1 for e in wrong_predictions if signal_name in e['signals'])

            correct_ratio = correct_activation / len(correct_predictions) if correct_predictions else 0
            wrong_ratio = wrong_activation / len(wrong_predictions) if wrong_predictions else 0

            # Значення важливості - різниця між активацією в правильних і неправильних передбаченнях
            signal_importance[signal_name] = correct_ratio - wrong_ratio

        # Оновлюємо ваги на основі важливості сигналів
        updated_weights = current_weights.copy()

        for signal_name, importance in signal_importance.items():
            # Збільшуємо або зменшуємо вагу залежно від важливості
            updated_weights[signal_name] = max(0.05,
                                               min(0.5, current_weights[signal_name] + self.learning_rate * importance))

        # Нормалізуємо ваги, щоб їх сума не перевищувала 1.5
        weight_sum = sum(updated_weights.values())
        if weight_sum > 1.5:
            scaling_factor = 1.5 / weight_sum
            updated_weights = {k: v * scaling_factor for k, v in updated_weights.items()}

        # Зберігаємо оновлені ваги
        self.token_weights[category] = updated_weights
        self._save_weights()

        print(
            f"Оновлено ваги для категорії {category}. Правильних: {len(correct_predictions)}, неправильних: {len(wrong_predictions)}")

    def _train_ml_model(self, category):
        """
        Навчання ML моделі для певної категорії токенів

        :param category: Категорія токена
        """
        # Отримуємо записи з відомими результатами
        labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]

        if len(labeled_entries) < self.retraining_interval:
            return

        # Підготовка даних
        X = np.array([e['features'] for e in labeled_entries])
        y = np.array([1 if e['actual_event'] else 0 for e in labeled_entries])

        # Стандартизація ознак
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Розділення на тренувальні та тестові дані
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Навчання моделі
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Оцінка моделі
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Навчено ML модель для категорії {category}:")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Зберігаємо модель
        self.ml_models[category] = model
        self._save_ml_model(category)

        # Очищаємо історію, залишаючи останні 50 записів для наступного навчання
        self.training_history[category] = self.training_history[category][-50:]

    def get_optimized_threshold(self, symbol):
        """
        Отримання оптимізованого порогу для класифікації

        :param symbol: Символ криптовалюти
        :return: Оптимізований поріг
        """
        category = self._get_token_category(symbol)

        # Базовий поріг
        base_currency = symbol.split('/')[0]
        if base_currency in ['PEPE', 'SHIB', 'DOGE', 'CTT']:
            base_threshold = 0.12  # Нижчий поріг для мем-токенів
        elif base_currency in ['TON', 'SOL', 'DOT', 'AVAX']:
            base_threshold = 0.15  # Середній поріг для альткоїнів
        else:
            base_threshold = 0.18  # Вищий поріг для інших токенів

        # Якщо є достатньо даних, обчислюємо оптимізований поріг
        if category in self.training_history:
            labeled_entries = [e for e in self.training_history[category] if e['actual_event'] is not None]

            if len(labeled_entries) >= 30:
                # Використовуємо F1-score для пошуку оптимального порогу
                thresholds = np.linspace(0.1, 0.5, 41)  # Від 0.1 до 0.5 з кроком 0.01
                best_f1 = 0
                best_threshold = base_threshold

                for threshold in thresholds:
                    y_true = [1 if e['actual_event'] else 0 for e in labeled_entries]
                    y_pred = [1 if e['probability'] > threshold else 0 for e in labeled_entries]

                    if sum(y_pred) > 0:  # Переконуємося, що є хоча б одне позитивне передбачення
                        f1 = f1_score(y_true, y_pred, zero_division=0)

                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold

                return best_threshold

        # Якщо не можемо обчислити оптимізований поріг, повертаємо базовий
        return base_threshold

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

        # Додаткові розрахунки для прискорення об'єму
        if not data_window.empty and len(data_window) >= 6:
            data_window['volume_change'] = data_window['volume'].pct_change()
            volume_acceleration = data_window['volume_change'].diff()[-5:].mean()
            volume_analysis['volume_acceleration'] = volume_acceleration
        else:
            volume_analysis['volume_acceleration'] = 0

        # Імітація аналізу книги ордерів
        order_book_analysis = {
            'order_book_signal': False,
            'buy_sell_ratio': 1.0
        }

        # Імітація соціальних даних
        social_data = {
            'social_signal': False,
            'mentions': 0,
            'average_mentions': 0,
            'percent_change': 0
        }

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

        # Отримуємо оптимізовані ваги
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

        # Сигнал 3: Прискорення об'єму
        if volume_analysis.get('volume_acceleration', 0) > 0.05:
            signals.append({
                'name': 'Прискорення зростання об\'єму',
                'description': f"Швидкість зростання об\'єму збільшується: {volume_analysis['volume_acceleration']:.2f}",
                'weight': weights.get('Прискорення зростання об\'єму', 0.25)
            })
            confidence += weights.get('Прискорення зростання об\'єму', 0.25) * min(
                volume_analysis['volume_acceleration'] / 0.15, 1.0)

        # Сигнал 4: Часовий патерн
        if time_pattern_data['time_pattern_signal']:
            signals.append({
                'name': 'Підозрілий часовий патерн',
                'description': f"Поточний час відповідає високоризиковому періоду для pump-and-dump схем",
                'weight': weights.get('Підозрілий часовий патерн', 0.15)
            })
            confidence += weights.get('Підозрілий часовий патерн', 0.15) * time_pattern_data['time_risk_score']

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
                'social': social_data
            }
        }

        return result
