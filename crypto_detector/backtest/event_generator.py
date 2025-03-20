import pandas as pd
import numpy as np


class EventGenerator:
    """
    Клас для генерації тестових подій на основі історичних даних.
    Ідентифікує pump-and-dump патерни та інші підозрілі події.
    """

    def __init__(self, min_price_change=5.0, min_volume_change=30.0, min_dump_percent=5.0):
        """
        Ініціалізація генератора подій

        :param min_price_change: Мінімальна зміна ціни в % для визначення pump-фази
        :param min_volume_change: Мінімальна зміна об'єму в % для визначення підвищеної активності
        :param min_dump_percent: Мінімальний відсоток падіння після піку для визначення dump-фази
        """
        self.min_price_change = min_price_change
        self.min_volume_change = min_volume_change
        self.min_dump_percent = min_dump_percent

    def generate_events(self, historical_data, window_hours=24):
        """
        Генерація тестових подій на основі історичних даних

        :param historical_data: DataFrame з історичними OHLCV даними
        :param window_hours: Вікно для аналізу в годинах
        :return: DataFrame з подіями
        """
        # Перевірка наявності необхідних колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in historical_data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {required_columns}")

        # Підготовка даних для аналізу
        data = historical_data.copy()

        # Визначення кількості свічок у вікні (залежить від таймфрейму даних)
        timeframe_minutes = self._estimate_timeframe_minutes(data)
        window_size = int(window_hours * 60 / timeframe_minutes)

        # Розрахунок додаткових індикаторів
        data['price_change'] = data['close'].pct_change(window_size) * 100
        data['volatility'] = data['close'].pct_change().rolling(window=int(window_size / 2)).std() * 100
        data['volume_change'] = data['volume'].pct_change(int(window_size / 2)) * 100

        # Знаходження точок з суттєвою зміною ціни
        events = []

        for i in range(window_size, len(data) - window_size):
            # Перевірка на PUMP фазу (різке зростання ціни)
            if data['price_change'].iloc[i] > self.min_price_change:
                # Перевірка наступних свічок на DUMP
                forward_window = data.iloc[i:i + window_size]
                if len(forward_window) >= window_size / 2:
                    max_price = forward_window['high'].max()
                    max_price_idx = forward_window['high'].idxmax()
                    end_price = forward_window['close'].iloc[-1]
                    dump_percent = (end_price / max_price - 1) * 100

                    # Додаткова перевірка на підвищений об'єм
                    volume_spike = data['volume_change'].iloc[i] > self.min_volume_change

                    # Визначення типу події
                    event_type = 'unknown'
                    if dump_percent < -self.min_dump_percent:
                        event_type = 'pump_and_dump'  # Класичний pump-and-dump
                    elif volume_spike and data['price_change'].iloc[i] > self.min_price_change * 1.5:
                        event_type = 'pump_only'  # Тільки pump з підвищеним об'ємом
                    else:
                        continue  # Пропускаємо події, які не відповідають критеріям

                    # Визначення початку події (момент, коли варто виявити підозрілу активність)
                    # Зазвичай за 24 години до початку pump-фази
                    event_start_idx = max(0, i - window_size)
                    event_start_time = data.index[event_start_idx]

                    # Формування даних події
                    event_data = {
                        'timestamp': event_start_time,
                        'start_price': data['close'].iloc[event_start_idx],
                        'peak_price': max_price,
                        'peak_time': max_price_idx,
                        'end_price': end_price,
                        'pump_percent': data['price_change'].iloc[i],
                        'dump_percent': dump_percent,
                        'volume_change': data['volume_change'].iloc[i],
                        'event_type': event_type,
                        'is_event': 1
                    }
                    events.append(event_data)

        # Створення DataFrame з подіями
        if events:
            events_df = pd.DataFrame(events)
            events_df.set_index('timestamp', inplace=True)
            return events_df
        else:
            return pd.DataFrame()

    def generate_non_events(self, historical_data, events_df, ratio=1.5):
        """
        Генерація точок, які не є подіями (для збалансованого навчання)

        :param historical_data: DataFrame з історичними OHLCV даними
        :param events_df: DataFrame з подіями
        :param ratio: Співвідношення не-подій до подій
        :return: DataFrame з не-подіями
        """
        if events_df.empty:
            return pd.DataFrame()

        # Більш ефективний алгоритм: використання векторизованих операцій
        all_timestamps = historical_data.index
        invalid_timestamps = set()  # Множина для швидкого пошуку

        # Визначення часового вікна для виключення (24 години до та після кожної події)
        for event_time in events_df.index:
            # Визначаємо часові межі
            start_time = event_time - pd.Timedelta(hours=24)
            end_time = event_time + pd.Timedelta(hours=24)

            # Знаходимо індекси, які потрапляють у заборонений діапазон
            mask = (all_timestamps >= start_time) & (all_timestamps <= end_time)
            invalid_indices = all_timestamps[mask]

            # Додаємо знайдені часові мітки до множини заборонених
            invalid_timestamps.update(invalid_indices)

        # Фільтруємо дійсні часові мітки
        valid_timestamps = [ts for ts in all_timestamps if ts not in invalid_timestamps]

        # Розрахунок кількості не-подій
        n_events = len(events_df)
        n_non_events = min(int(n_events * ratio), len(valid_timestamps))

        # Випадковий вибір часових міток для не-подій
        if n_non_events > 0 and valid_timestamps:
            # Випадковий вибір часових міток
            non_event_timestamps = np.random.choice(valid_timestamps, size=n_non_events, replace=False)

            # Формування даних не-подій (використовуємо більш ефективний метод)
            non_events = []
            for timestamp in non_event_timestamps:
                try:
                    idx = historical_data.index.get_loc(timestamp)
                    non_event_data = {
                        'timestamp': timestamp,
                        'start_price': historical_data['close'].iloc[idx],
                        'event_type': 'non_event',
                        'is_event': 0
                    }
                    non_events.append(non_event_data)
                except (KeyError, IndexError):
                    # Пропускаємо, якщо індекс не знайдено
                    continue

            # Створення DataFrame з не-подіями
            if non_events:  # Перевіряємо, чи не порожній список
                non_events_df = pd.DataFrame(non_events)
                non_events_df.set_index('timestamp', inplace=True)
                return non_events_df

        return pd.DataFrame()

    def combine_events(self, events_df, non_events_df):
        """
        Об'єднання подій та не-подій для створення збалансованого набору даних

        :param events_df: DataFrame з подіями
        :param non_events_df: DataFrame з не-подіями
        :return: DataFrame з усіма подіями
        """
        if events_df.empty and non_events_df.empty:
            return pd.DataFrame()
        elif events_df.empty:
            return non_events_df
        elif non_events_df.empty:
            return events_df
        else:
            # Об'єднання DataFrame
            combined_df = pd.concat([events_df, non_events_df])
            combined_df.sort_index(inplace=True)
            return combined_df

    def _estimate_timeframe_minutes(self, data):
        """
        Оцінка часового інтервалу даних у хвилинах

        :param data: DataFrame з історичними даними
        :return: Оцінка часового інтервалу у хвилинах
        """
        if len(data) < 2:
            return 5  # За замовчуванням припускаємо 5-хвилинний таймфрейм

        # Розрахунок медіанного інтервалу між сусідніми свічками
        time_diffs = []
        for i in range(1, min(100, len(data))):
            diff = (data.index[i] - data.index[i - 1]).total_seconds() / 60
            time_diffs.append(diff)

        # Використання медіани для стійкості до викидів
        median_diff = np.median(time_diffs)

        # Округлення до найближчого стандартного таймфрейму
        standard_timeframes = [1, 3, 5, 15, 30, 60, 240, 1440]
        return min(standard_timeframes, key=lambda x: abs(x - median_diff))

    def visualize_events(self, historical_data, events_df, n_samples=5):
        """
        Візуалізація виявлених подій

        :param historical_data: DataFrame з історичними OHLCV даними
        :param events_df: DataFrame з подіями
        :param n_samples: Кількість подій для візуалізації
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if events_df.empty:
            print("Немає подій для візуалізації")
            return

        # Вибір випадкових подій для візуалізації
        if len(events_df) > n_samples:
            samples = events_df.sample(n_samples)
        else:
            samples = events_df

        for idx, event in samples.iterrows():
            event_time = idx
            event_type = event.get('event_type', 'unknown')

            # Визначення вікна для візуалізації (48 годин до та 48 годин після події)
            start_time = event_time - pd.Timedelta(hours=48)
            end_time = event_time + pd.Timedelta(hours=48)

            # Вибір даних для вікна
            window_data = historical_data[
                (historical_data.index >= start_time) &
                (historical_data.index <= end_time)
                ]

            if window_data.empty:
                print(f"Недостатньо даних для візуалізації події на {event_time}")
                continue

            # Створення графіка
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

            # Графік ціни
            ax1.plot(window_data.index, window_data['close'], label='Ціна закриття')

            # Позначення часу події
            ax1.axvline(x=event_time, color='r', linestyle='-', alpha=0.5, label='Час події')

            # Позначення піку (якщо є)
            if 'peak_time' in event and pd.notna(event['peak_time']):
                ax1.axvline(x=event['peak_time'], color='g', linestyle='--', alpha=0.5, label='Пік')

            ax1.set_title(f'Подія типу "{event_type}" на {event_time}')
            ax1.set_ylabel('Ціна')
            ax1.legend()
            ax1.grid(True)

            # Графік об'єму
            ax2.bar(window_data.index, window_data['volume'], alpha=0.7, label='Об\'єм')
            ax2.axvline(x=event_time, color='r', linestyle='-', alpha=0.5)
            ax2.set_ylabel('Об\'єм')
            ax2.legend()
            ax2.grid(True)

            # Форматування осі X
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.show()
