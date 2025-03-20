from sklearn.ensemble import IsolationForest


class VolumeAnalyzer:
    """
    Клас для аналізу об'єму торгів криптовалют.
    Виявляє аномальні об'єми та розраховує пов'язані метрики.
    """

    def __init__(self, threshold_multiplier=2):
        """
        Ініціалізація аналізатора об'єму

        :param threshold_multiplier: Множник для визначення незвичайної активності
        """
        self.threshold_multiplier = threshold_multiplier

    async def detect_unusual_volume(self, df_ohlcv):
        """
        Виявлення незвичайних обсягів торгів

        :param df_ohlcv: DataFrame з OHLCV даними
        :return: Dict з результатами аналізу
        """
        if df_ohlcv.empty:
            return {'unusual_volume': False, 'z_score': 0, 'anomaly_count': 0}

        # Відділяємо історичні дані від найновіших для порівняння
        historical = df_ohlcv[:-5]  # Всі дані крім останніх 5 свічок
        recent = df_ohlcv[-5:]  # Останні 5 свічок

        if historical.empty or recent.empty:
            return {'unusual_volume': False, 'z_score': 0, 'anomaly_count': 0}

        # Розрахунок статистичних показників
        mean_volume = historical['volume'].mean()
        std_volume = historical['volume'].std()
        recent_volume = recent['volume'].mean()

        if std_volume == 0:  # Запобігаємо діленню на нуль
            z_score = 0
        else:
            z_score = (recent_volume - mean_volume) / std_volume

        # Розрахунок швидкості зміни об'єму
        df_volume = df_ohlcv.copy()
        df_volume['volume_change'] = df_volume['volume'].pct_change()
        recent_volume_change = df_volume['volume_change'][-5:].mean() if not df_volume.empty else 0
        volume_acceleration = df_volume['volume_change'].diff()[-5:].mean() if len(df_volume) >= 6 else 0

        # Використовуємо Isolation Forest для виявлення аномалій
        recent_anomalies = 0
        if len(df_ohlcv) > 10:  # Потрібно достатньо даних для алгоритму
            clf = IsolationForest(contamination=0.1, random_state=42)
            volume_data = df_ohlcv[['volume']].copy()
            volume_data['anomaly'] = clf.fit_predict(volume_data)
            recent_anomalies = (volume_data['anomaly'][-5:] == -1).sum()

        return {
            'unusual_volume': z_score > self.threshold_multiplier or recent_anomalies >= 2 or recent_volume_change > 0.5,
            'z_score': z_score,
            'anomaly_count': recent_anomalies,
            'recent_volume': recent_volume,
            'mean_volume': mean_volume,
            'percent_change': ((recent_volume / mean_volume) - 1) * 100 if mean_volume > 0 else 0,
            'volume_acceleration': volume_acceleration,
            'volume_change_rate': recent_volume_change
        }

    async def analyze_historical_volume(self, data_window):
        """
        Аналіз історичного об'єму з менш жорсткими пороговими значеннями

        :param data_window: Вікно даних для аналізу
        :return: Результати аналізу об'єму
        """
        if data_window.empty or len(data_window) < 10:
            return {'unusual_volume': False, 'z_score': 0, 'percent_change': 0}

        # Відділяємо історичні дані від найновіших для порівняння
        historical = data_window[:-5]  # Всі дані крім останніх 5 свічок
        recent = data_window[-5:]  # Останні 5 свічок

        if historical.empty or recent.empty:
            return {'unusual_volume': False, 'z_score': 0, 'percent_change': 0}

        # Розрахунок статистичних показників
        mean_volume = historical['volume'].mean()
        std_volume = historical['volume'].std()
        recent_volume = recent['volume'].mean()

        if std_volume == 0:  # Запобігаємо діленню на нуль
            z_score = 0
        else:
            z_score = (recent_volume - mean_volume) / std_volume

        percent_change = ((recent_volume / mean_volume) - 1) * 100 if mean_volume > 0 else 0

        # Розрахунок зміни об'єму для останніх свічок
        volume_changes = data_window['volume'].pct_change().dropna()
        recent_changes = volume_changes[-5:] if len(volume_changes) >= 5 else volume_changes
        max_recent_change = recent_changes.max() * 100 if not recent_changes.empty else 0

        return {
            'unusual_volume': z_score > 2 or percent_change > 30 or max_recent_change > 50,
            'z_score': z_score,
            'recent_volume': recent_volume,
            'mean_volume': mean_volume,
            'percent_change': percent_change,
            'max_recent_change': max_recent_change
        }