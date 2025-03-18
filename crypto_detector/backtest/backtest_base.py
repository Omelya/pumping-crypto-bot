import os
import json
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


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
        base_currency = symbol.split('/')[0]
        if base_currency in ['PEPE', 'SHIB', 'DOGE', 'CTT']:
            return 0.12  # Нижчий поріг для мем-токенів
        elif base_currency in ['TON', 'SOL', 'DOT', 'AVAX']:
            return 0.15  # Середній поріг для альткоїнів
        else:
            return 0.18  # Вищий поріг для інших токенів

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
            filename = os.path.join(symbol_dir, f"currency/{timeframe}_{start_date}_{end_date}.csv")
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

    def generate_test_events(self, historical_data, min_price_change=5.0, window=24):
        """
        Генерація тестових подій на основі історичних даних

        :param historical_data: DataFrame з історичними даними
        :param min_price_change: Мінімальна зміна ціни в % для позначення як події
        :param window: Вікно для аналізу в годинах
        :return: DataFrame з подіями
        """
        # Розрахунок % зміни за період
        historical_data['price_change'] = historical_data['close'].pct_change(window) * 100

        # Додавання волатильності та об'єму для аналізу
        historical_data['volatility'] = historical_data['close'].pct_change().rolling(window=12).std() * 100
        historical_data['volume_change'] = historical_data['volume'].pct_change(12) * 100

        # Знаходження точок з суттєвою зміною ціни
        events = []

        for i in range(window, len(historical_data) - window):
            # Перевірка на PUMP фазу (різке зростання ціни)
            if historical_data['price_change'].iloc[i] > min_price_change:
                # Перевірка наступних свічок на DUMP
                forward_window = historical_data.iloc[i:i + window]
                if len(forward_window) >= 10:
                    max_price = forward_window['high'].max()
                    max_price_idx = forward_window['high'].idxmax()
                    end_price = forward_window['close'].iloc[-1]
                    dump_percent = (end_price / max_price - 1) * 100

                    # Додаткова перевірка на підвищений об'єм
                    volume_spike = historical_data['volume_change'].iloc[i] > 30  # 30% зростання об'єму

                    # Якщо відбувся і pump і dump, або просто сильний pump з підвищеним об'ємом
                    if dump_percent < -5 or (
                            volume_spike and historical_data['price_change'].iloc[i] > min_price_change * 1.5):
                        event_time = historical_data.index[i - window]
                        event_data = {
                            'timestamp': event_time,
                            'start_price': historical_data['close'].iloc[i - window],
                            'peak_price': max_price,
                            'peak_time': max_price_idx,
                            'end_price': end_price,
                            'pump_percent': historical_data['price_change'].iloc[i],
                            'dump_percent': dump_percent,
                            'volume_change': historical_data['volume_change'].iloc[i],
                            'is_event': 1
                        }
                        events.append(event_data)

        # Створення DataFrame з подіями
        if events:
            events_df = pd.DataFrame(events)
            events_df.set_index('timestamp', inplace=True)
            print(f"Знайдено {len(events_df)} pump-and-dump подій")
            return events_df
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

        # Додавання випадкових точок, які не є подіями (з кращим балансом)
        non_event_times = historical_data.index.difference(events_df.index)
        if len(non_event_times) > len(events_df):
            # Для кращого балансу класів
            balance_ratio = 1.5  # Зменшено для більш збалансованого набору даних
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

            print(
                f"Аналіз на {timestamp}: Подія={is_event}, Прогноз={prediction}, Ймовірність={result['probability_score']:.2f}")

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

        # Сигнал 3: Прискорення об'єму
        if volume_analysis.get('volume_acceleration', 0) > 0.05:
            signals.append({
                'name': 'Прискорення зростання об\'єму',
                'description': f"Швидкість зростання об\'єму збільшується: {volume_analysis['volume_acceleration']:.2f}",
                'weight': 0.25
            })
            confidence += 0.25 * min(volume_analysis['volume_acceleration'] / 0.15, 1.0)

        # Сигнал 4: Часовий патерн
        if time_pattern_data['time_pattern_signal']:
            signals.append({
                'name': 'Підозрілий часовий патерн',
                'description': f"Поточний час відповідає високоризиковому періоду для pump-and-dump схем",
                'weight': 0.15
            })
            confidence += 0.15 * time_pattern_data['time_risk_score']

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

    def save_backtest_results(self, filename="backtest_results.json"):
        """
        Збереження результатів бектестингу у файл

        :param filename: Ім'я файлу для збереження
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        print(f"Результати бектестингу збережено у {filename}")

    def visualize_results(self, symbol):
        """
        Візуалізація результатів бектестингу

        :param symbol: Символ криптовалюти
        """
        if symbol not in self.results:
            print(f"Немає результатів бектестингу для {symbol}")
            return

        results = self.results[symbol]

        # Підготовка даних для візуалізації
        predictions_df = pd.DataFrame(results['predictions'])
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df.set_index('timestamp', inplace=True)
        predictions_df.sort_index(inplace=True)

        # Створення директорії для графіків, якщо вона не існує
        charts_dir = os.path.join(self.data_dir, "charts")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        # Візуалізація результатів
        plt.figure(figsize=(15, 10))

        # Графік 1: Розподіл ймовірностей
        plt.subplot(2, 2, 1)
        sns.histplot(predictions_df['probability'], bins=20, kde=True)
        plt.title(f'Розподіл ймовірностей для {symbol}')
        plt.xlabel('Ймовірність')
        plt.ylabel('Кількість')

        # Графік 2: ROC-крива (спрощено)
        plt.subplot(2, 2, 2)
        plt.scatter(predictions_df[predictions_df['is_event'] == 0]['probability'],
                    np.zeros(len(predictions_df[predictions_df['is_event'] == 0])),
                    color='blue', alpha=0.5, label='Не події')
        plt.scatter(predictions_df[predictions_df['is_event'] == 1]['probability'],
                    np.ones(len(predictions_df[predictions_df['is_event'] == 1])),
                    color='red', alpha=0.5, label='Події')
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
        # Додаємо лінію порогу класифікації
        threshold = self.get_prediction_threshold(symbol)
        plt.axvline(x=threshold, color='g', linestyle='--', alpha=0.7, label=f'Поріг = {threshold}')
        plt.title('Розподіл ймовірностей для подій та не-подій')
        plt.xlabel('Ймовірність')
        plt.ylabel('Клас')
        plt.legend()

        # Графік 3: Часова динаміка ймовірностей
        plt.subplot(2, 1, 2)
        plt.plot(predictions_df.index, predictions_df['probability'], 'b-', label='Ймовірність')

        # Позначаємо реальні події
        events = predictions_df[predictions_df['is_event'] == 1]
        if not events.empty:
            plt.scatter(events.index, events['probability'], color='red', s=50, label='Реальні події')

        # Позначаємо передбачені події
        predicted = predictions_df[predictions_df['prediction'] == 1]
        if not predicted.empty:
            plt.scatter(predicted.index, predicted['probability'], marker='x', color='green', s=50,
                        label='Передбачені події')

        # Лінія порогу
        plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.7)

        plt.title(f'Часова динаміка ймовірностей для {symbol}')
        plt.xlabel('Дата')
        plt.ylabel('Ймовірність')
        plt.legend()

        plt.tight_layout()

        # Збереження графіка у вказаній директорії
        chart_file = os.path.join(charts_dir, f"{symbol.replace('/', '_')}_backtest_results.png")
        plt.savefig(chart_file)
        plt.close()  # Закриваємо фігуру для звільнення пам'яті

        print(f"Візуалізацію збережено як {chart_file}")
