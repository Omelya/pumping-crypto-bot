import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from crypto_detector.backtest.event_generator import EventGenerator


class CryptoBacktester:
    """
    Клас для проведення бектестування на історичних даних.
    Дозволяє оцінити ефективність алгоритму виявлення pump-and-dump схем.
    """

    def __init__(self, detector, data_dir="historical_data"):
        """
        Ініціалізація бектестера

        :param detector: Екземпляр CryptoActivityDetector
        :param data_dir: Директорія з історичними даними
        """
        self.detector = detector
        self.data_dir = data_dir
        self.results = {}

        # Створення директорії для даних, якщо вона не існує
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def get_prediction_threshold(self, symbol):
        """
        Динамічний поріг для різних токенів

        :param symbol: Символ криптовалюти
        :return: Поріг для класифікації
        """
        from crypto_detector.config.settings import TOKEN_THRESHOLDS, TOKEN_CATEGORIES

        base_currency = symbol.split('/')[0]

        # Визначення категорії токена
        category = 'other'  # За замовчуванням
        for cat, tokens in TOKEN_CATEGORIES.items():
            if base_currency in tokens:
                category = cat
                break

        # Повернення порогу на основі категорії
        return TOKEN_THRESHOLDS.get(category, TOKEN_THRESHOLDS['other'])

    async def download_historical_data(self, symbol, start_date, end_date, timeframe='5m'):
        """
        Завантаження історичних даних з біржі

        :param symbol: Символ криптовалюти (наприклад, 'BTC/USDT')
        :param start_date: Початкова дата у форматі 'YYYY-MM-DD'
        :param end_date: Кінцева дата у форматі 'YYYY-MM-DD'
        :param timeframe: Інтервал часу для даних
        :return: Шлях до збереженого файлу з даними
        """
        print(f"Завантаження історичних даних для {symbol} з {start_date} по {end_date}...")

        # Конвертація дат в об'єкти datetime
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

        # Створення директорії для символу, якщо вона не існує
        symbol_dir = os.path.join(self.data_dir, 'currency', symbol.replace('/', '_'))
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)

        # Отримання даних OHLCV
        all_ohlcv = []
        current_date = start_datetime

        while current_date <= end_datetime:
            next_date = current_date + timedelta(days=1)

            try:
                # Конвертація timestamp в мілісекунди для CCXT
                since = int(current_date.timestamp() * 1000)

                # Отримання OHLCV даних за один день
                ohlcv = self.detector.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=1000  # Максимальна кількість свічок
                )

                if ohlcv:
                    all_ohlcv.extend(ohlcv)
                    print(f"  Отримано {len(ohlcv)} свічок для {current_date.strftime('%Y-%m-%d')}")

                # Затримка, щоб не перевищити ліміти API
                await asyncio.sleep(1)

            except Exception as e:
                print(f"  Помилка завантаження даних для {current_date.strftime('%Y-%m-%d')}: {str(e)}")

            current_date = next_date

        # Конвертування у DataFrame
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Збереження даних у файл
            filename = os.path.join(symbol_dir, f"{timeframe}_{start_date}_{end_date}.csv")
            df.to_csv(filename)
            print(f"Збережено {len(df)} записів у {filename}")

            return filename
        else:
            print("Не вдалося отримати дані")
            return None

    def load_historical_data(self, filename):
        """
        Завантаження збережених історичних даних з файлу

        :param filename: Шлях до файлу з даними
        :return: DataFrame з даними
        """
        try:
            df = pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
            return df
        except Exception as e:
            print(f"Помилка завантаження даних з {filename}: {str(e)}")
            return None

    def generate_test_events(self, historical_data, min_price_change=5.0, window=24, generate_non_events=True,
                             visualization=False):
        """
        Генерація тестових подій на основі історичних даних

        :param historical_data: DataFrame з історичними даними
        :param min_price_change: Мінімальна зміна ціни в % для позначення як події
        :param window: Вікно для аналізу в годинах
        :param generate_non_events: Чи генерувати не-події для збалансованого навчання
        :param visualization: Чи візуалізувати знайдені події
        :return: DataFrame з подіями
        """
        # Використання EventGenerator для виявлення подій
        event_generator = EventGenerator(
            min_price_change=min_price_change,
            min_volume_change=30.0,  # Використовуємо стандартне значення з вашого коду
            min_dump_percent=15.0  # Використовуємо стандартне значення з вашого коду
        )

        # Генерація подій за допомогою EventGenerator
        events_df = event_generator.generate_events(historical_data, window_hours=window)

        if not events_df.empty:
            print(f"До фільтрації: {len(events_df)} подій")
            print(f"- Pump-and-dump: {len(events_df[events_df['event_type'] == 'pump_and_dump'])}")
            print(f"- Pump only: {len(events_df[events_df['event_type'] == 'pump_only'])}")

            # 1. Залишаємо тільки події з найвищим pump_percent (верхні 25%)
            pump_threshold = events_df['pump_percent'].quantile(0.75)
            events_df = events_df[events_df['pump_percent'] >= pump_threshold]

            # 2. Переконуємося, що події розділені в часі (мінімум 12 годин між подіями)
            events_to_keep = []
            events_sorted = events_df.sort_index()
            last_event_time = None

            for event_time, event in events_sorted.iterrows():
                if last_event_time is None or (event_time - last_event_time).total_seconds() / 3600 >= 12:
                    events_to_keep.append(event_time)
                    last_event_time = event_time

            filtered_events_df = events_df.loc[events_to_keep]

            # Виводимо інформацію про відфільтровані події
            print(f"Після фільтрації: {len(filtered_events_df)} подій")
            print(f"- Pump-and-dump: {len(filtered_events_df[filtered_events_df['event_type'] == 'pump_and_dump'])}")
            print(f"- Pump only: {len(filtered_events_df[filtered_events_df['event_type'] == 'pump_only'])}")

            # Використовуємо відфільтровані події
            events_df = filtered_events_df

            # Для сумісності з існуючим кодом забезпечуємо наявність 'is_event' колонки
            if 'is_event' not in events_df.columns:
                events_df['is_event'] = 1

            # Генерація не-подій, якщо вказано
            combined_df = events_df.copy()

            if generate_non_events:
                non_events_df = event_generator.generate_non_events(historical_data, events_df, ratio=1.5)

                if not non_events_df.empty:
                    combined_df = event_generator.combine_events(events_df, non_events_df)
                    events_count = len(combined_df[combined_df['is_event'] == 1])
                    non_events_count = len(combined_df[combined_df['is_event'] == 0])
                    print(f"\nСтворено набір даних з {events_count} подій та {non_events_count} не-подій")
                    print(f"Співвідношення: {non_events_count / events_count:.2f}")

            # Візуалізація подій, якщо вказано
            if visualization:
                event_generator.visualize_events(historical_data, events_df, n_samples=3)

            return combined_df
        else:
            print("Не знайдено подій, що відповідають критеріям")
            return pd.DataFrame()

    async def backtest_algorithm(self, symbol, start_date, end_date, min_price_change=5.0, window_hours=24):
        """
        Бектестинг алгоритму на історичних даних

        :param symbol: Символ криптовалюти
        :param start_date: Початкова дата
        :param end_date: Кінцева дата
        :param min_price_change: Мінімальна зміна ціни для позначення події
        :param window_hours: Вікно для аналізу в годинах
        :return: Результати бектестингу
        """
        # Шлях до файлу з історичними даними
        data_file = os.path.join(
            self.data_dir,
            'currency',
            symbol.replace('/', '_'),
            f"5m_{start_date}_{end_date}.csv"
        )

        # Перевірка, чи існують дані, якщо ні - завантажуємо
        if not os.path.exists(data_file):
            data_file = await self.download_historical_data(symbol, start_date, end_date)
            if not data_file:
                print("Не вдалося отримати історичні дані для бектестингу")
                return None

        # Завантаження історичних даних
        historical_data = self.load_historical_data(data_file)

        if historical_data is None:
            return None

        # Генерація тестових подій
        events_df = self.generate_test_events(historical_data, min_price_change,
                                              window=int(window_hours * 12))  # 12 5-хвилинних свічок на годину
        if events_df.empty:
            print("Не знайдено тестових подій")
            return None

        print(f"Знайдено {len(events_df)} тестових подій")

        # Підготовка для бектестингу
        test_timestamps = []

        # Додавання подій як тестових точок
        for event_time in events_df.index:
            test_timestamps.append(event_time)

        non_event_times = historical_data.index.difference(events_df.index)

        if len(non_event_times) > len(events_df):
            balance_ratio = 1.5

            random_non_events = np.random.choice(non_event_times,
                                                 size=min(int(len(events_df) * balance_ratio), len(non_event_times)),
                                                 replace=False)

            test_timestamps.extend(random_non_events)

        # Бектестинг на кожній тестовій точці
        y_true = []
        y_pred = []
        predictions = []

        # Динамічний поріг для класифікації
        prediction_threshold = self.get_prediction_threshold(symbol)
        print(f"Використовується поріг класифікації {prediction_threshold} для {symbol}")

        for timestamp in test_timestamps:
            # Визначення чи є ця точка подією
            is_event = 1 if timestamp in events_df.index else 0
            y_true.append(is_event)

            # Підготовка даних для аналізу
            cutoff_time = timestamp
            data_window = historical_data[:cutoff_time].tail(int(window_hours * 12))  # Дані до цього моменту

            # Перевірка наявності достатньої кількості даних
            if len(data_window) < window_hours:
                print(f"Недостатньо даних для аналізу на {timestamp}")
                y_pred.append(0)
                continue

            # Виклик методу аналізу з історичними даними
            result = await self._analyze_historical_point(self.detector, symbol, data_window, timestamp)

            # Визначення передбачення з динамічним порогом
            prediction = 1 if result['probability_score'] > prediction_threshold else 0
            y_pred.append(prediction)

            # Збереження деталей
            predictions.append({
                'timestamp': pd.Timestamp(timestamp).isoformat(),
                'is_event': is_event,
                'prediction': prediction,
                'probability': result['probability_score'],
                'signals': result['signals']
            })

        # Розрахунок метрик
        if len(y_true) > 0 and len(y_pred) > 0:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            # Збереження результатів
            backtest_results = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'predictions': predictions
            }

            self.results[symbol] = backtest_results

            # Виведення результатів
            print("\nРезультати бектестингу:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)

            # Створення директорії для зберігання результатів ML, якщо вона не існує
            ml_data_dir = os.path.join(self.data_dir, "ml_data")
            if not os.path.exists(ml_data_dir):
                os.makedirs(ml_data_dir)

            # Підготовка даних для ML
            ml_data = pd.DataFrame(predictions)
            ml_data_file = os.path.join(ml_data_dir, f"{symbol.replace('/', '_')}_ml_data.csv")
            ml_data.to_csv(ml_data_file, index=False)
            print(f"Дані для машинного навчання збережені у {ml_data_file}")

            return backtest_results
        else:
            print("Не вдалося виконати бектестинг через недостатню кількість даних")
            return None

    async def _analyze_historical_point(self, detector, symbol, data_window, timestamp):
        """
        Аналіз історичної точки даних

        :param detector: Екземпляр CryptoActivityDetector
        :param symbol: Символ криптовалюти
        :param data_window: Вікно історичних даних для аналізу
        :param timestamp: Часова мітка для аналізу
        :return: Результат аналізу
        """
        # Аналіз об'єму
        volume_analysis = await detector.volume_analyzer.analyze_historical_volume(data_window)

        # Аналіз цінової динаміки
        price_analysis = await detector.price_analyzer.analyze_historical_price(data_window)

        # Виявлення нових патернів
        vertical_jump, jump_percent = detector.price_analyzer.detect_vertical_price_jump(data_window)
        v_pattern_detected = detector.price_analyzer.detect_v_pattern(data_window)
        large_green_candle, candle_body_percent = detector.price_analyzer.detect_large_candles(data_window)

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

        order_book_analysis = await detector.orderbook_analyzer._analyze_historical_orderbook(timestamp, data_window)

        social_data = {
            'social_signal': False,
            'mentions': 0,
            'average_mentions': 0,
            'percent_change': 0
        }

        # Аналіз кореляції на історичних даних
        correlation_data = await self._analyze_historical_correlation(
            detector, symbol, timestamp, data_window
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

        # Розрахунок оцінки ймовірності
        signals = []
        confidence = 0.0

        # Сигнал 1: Аномальний обсяг торгів
        if volume_analysis['unusual_volume']:
            signals.append({
                'name': 'Аномальний обсяг торгів',
                'description': f"Поточний обсяг перевищує середній на {volume_analysis['percent_change']:.2f}%",
                'weight': 0.35
            })
            confidence += 0.35 * min(volume_analysis['percent_change'] / 80, 1.0) if volume_analysis[
                                                                                         'percent_change'] > 0 else 0

        # Сигнал 2: Цінова динаміка
        if price_analysis['price_action_signal']:
            signals.append({
                'name': 'Активна цінова динаміка',
                'description': f"Зміна ціни за останній період: {price_analysis['recent_price_change']:.2f}%",
                'weight': 0.25
            })
            confidence += 0.25 * min(abs(price_analysis['recent_price_change']) / 8, 1.0)

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
                # ratio < 1 і не дорівнює 0
                confidence += 0.2 * min(1 / ratio / 1.7, 1.0)

        # Сигнал 5: Часовий патерн
        if time_pattern_data['time_pattern_signal']:
            signals.append({
                'name': 'Підозрілий часовий патерн',
                'description': f"Поточний час відповідає високоризиковому періоду для pump-and-dump схем",
                'weight': 0.15
            })
            confidence += 0.15 * time_pattern_data['time_risk_score']

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
        if volume_analysis.get('volume_acceleration', 0) > 0.1:
            signals.append({
                'name': 'Прискорення зростання об\'єму',
                'description': f"Швидкість зростання об\'єму збільшується: {volume_analysis['volume_acceleration']:.2f}",
                'weight': 0.15
            })
            confidence += 0.15 * min(volume_analysis['volume_acceleration'] / 0.2, 1.0)

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
                'weight': 0.40
            })
            confidence += 0.40 * min(price_analysis.get('jump_percent', 0) / 50, 1.0)

        # Новий сигнал 11: Паттерн V-подібного руху ціни
        if price_analysis.get('v_pattern_detected', False):
            signals.append({
                'name': 'V-подібний патерн ціни',
                'description': f"Виявлено швидке зростання і падіння ціни без консолідації",
                'weight': 0.35
            })
            confidence += 0.35

        # Новий сигнал 12: Велика зелена свічка з довгим тілом
        if price_analysis.get('large_green_candle', False):
            signals.append({
                'name': 'Велика зелена свічка з довгим тілом',
                'description': f"Тіло свічки складає {price_analysis.get('candle_body_percent', 0):.2f}% від ціни",
                'weight': 0.30
            })
            confidence += 0.30 * min(price_analysis.get('candle_body_percent', 0) / 15, 1.0)

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
            # У випадку помилки повертаємо базовий результат
            print(f"Помилка аналізу історичної кореляції: {e}")
            return {
                'correlation_signal': False,
                'correlated_coins': [],
                'correlation_type': 'normal'
            }

    def save_backtest_results(self, filename="backtest_results.json"):
        """
        Збереження результатів бектестингу у файл

        :param filename: Ім'я файлу для збереження
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        print(f"Результати бектестингу збережено у {filename}")
