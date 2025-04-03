import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
import sqlite3
import os
import crypto_detector.config.settings as settings

class CorrelationAnalyzer:
    """
    Клас для аналізу кореляції між різними криптовалютами.
    Виявляє скоординовані pump-and-dump атаки, які часто охоплюють кілька монет одночасно.
    Зберігає дані в SQLite базі даних для обміну між різними екземплярами програми.
    """

    def __init__(self, window_size=24, correlation_threshold=0.75, pump_threshold=10.0):
        """
        Ініціалізація аналізатора кореляції

        :param window_size: Розмір вікна для аналізу (години)
        :param correlation_threshold: Поріг кореляції для виявлення пов'язаних монет
        :param pump_threshold: Поріг для визначення різкого зростання ціни (відсотки)
        """
        self.db_path = settings.DB_NAME
        self.window_size = window_size
        self.correlation_threshold = correlation_threshold
        self.pump_threshold = pump_threshold
        self.correlation_matrix = None
        self.last_matrix_update = None
        self.update_interval = timedelta(minutes=15)

        # Кеш для даних, отриманих з бази
        self.market_data_cache = {}
        self.cache_expiry = {}
        self.cache_lifetime = timedelta(minutes=5)

        # Ініціалізуємо базу даних
        self._init_database()

    def _init_database(self):
        """
        Ініціалізація бази даних
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Створюємо таблицю для ринкових даних, якщо вона не існує
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT,
            timestamp DATETIME,
            price REAL,
            volume REAL,
            PRIMARY KEY (symbol, timestamp)
        )
        ''')

        # Створюємо таблицю для кешування кореляційної матриці
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS correlation_matrix (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME,
            matrix_data TEXT
        )
        ''')

        # Створюємо індекси для покращення продуктивності
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data (timestamp)')

        conn.commit()
        conn.close()

    def _get_dataframe_for_symbol(self, symbol, hours=None):
        """
        Перетворення даних для конкретного символу у DataFrame

        :param symbol: Символ криптовалюти
        :param hours: Кількість годин історії для аналізу (якщо None, використовується window_size)
        :return: DataFrame з даними або None, якщо недостатньо даних
        """
        if hours is None:
            hours = self.window_size

        # Спочатку перевіряємо кеш
        cache_key = f"{symbol}_{hours}"
        now = datetime.now()

        if (cache_key in self.cache_expiry and
                now < self.cache_expiry[cache_key] and
                cache_key in self.market_data_cache):
            return self.market_data_cache[cache_key]

        # Обчислюємо часову межу
        cutoff_time = now - timedelta(hours=hours)

        # Отримуємо дані з бази
        conn = sqlite3.connect(self.db_path)
        query = '''
        SELECT timestamp, price, volume 
        FROM market_data 
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp
        '''

        try:
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, cutoff_time.isoformat()),
                parse_dates=['timestamp']
            )
            conn.close()

            if df.empty or len(df) < 5:
                return None

            # Встановлюємо timestamp як індекс
            df.set_index('timestamp', inplace=True)

            # Ресемплінг даних для отримання рівномірних часових інтервалів
            df = df.resample('5T').last().dropna()

            # Додаємо зміну ціни у відсотках
            df['pct_change'] = df['price'].pct_change() * 100

            # Кешуємо результат
            self.market_data_cache[cache_key] = df
            self.cache_expiry[cache_key] = now + self.cache_lifetime

            return df
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            conn.close()
            return None

    def _update_correlation_matrix(self):
        """
        Оновлення кореляційної матриці для всіх монет
        """
        now = datetime.now()

        # Перевіряємо, чи є збережена матриця в базі даних
        if self.last_matrix_update is None:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            SELECT timestamp, matrix_data FROM correlation_matrix
            ORDER BY timestamp DESC LIMIT 1
            ''')
            result = cursor.fetchone()
            conn.close()

            if result:
                timestamp_str, matrix_data = result
                timestamp = datetime.fromisoformat(timestamp_str)

                # Якщо матриця досить нова, використовуємо її
                if now - timestamp < self.update_interval:
                    self.correlation_matrix = pd.read_json(matrix_data)
                    self.last_matrix_update = timestamp
                    return

        # Перевіряємо, чи потрібно оновлювати матрицю
        if (self.last_matrix_update is not None and
                now - self.last_matrix_update < self.update_interval):
            return

        # Отримуємо список усіх символів у базі даних
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_data")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(symbols) < 2:
            return

        # Збираємо дані відсоткової зміни для всіх монет
        price_changes = {}

        for symbol in symbols:
            df = self._get_dataframe_for_symbol(symbol)
            if df is not None and 'pct_change' in df.columns and len(df) > 5:
                price_changes[symbol] = df['pct_change'].dropna()

        if len(price_changes) < 2:
            return

        # Створюємо пари символів і обчислюємо кореляцію між ними
        correlation_data = {}

        for i, symbol1 in enumerate(price_changes.keys()):
            correlation_data[symbol1] = {}

            for symbol2 in price_changes.keys():
                if symbol1 == symbol2:
                    correlation_data[symbol1][symbol2] = 1.0
                    continue

                # Знаходимо спільний проміжок часу
                s1 = price_changes[symbol1]
                s2 = price_changes[symbol2]

                # Перевіряємо, що є достатньо даних
                if len(s1) < 5 or len(s2) < 5:
                    correlation_data[symbol1][symbol2] = 0.0
                    continue

                try:
                    # Вирівнюємо часові ряди
                    common_index = s1.index.intersection(s2.index)
                    if len(common_index) < 5:
                        correlation_data[symbol1][symbol2] = 0.0
                        continue

                    # Обчислюємо кореляцію Пірсона
                    s1_aligned = s1.loc[common_index]
                    s2_aligned = s2.loc[common_index]
                    corr, _ = pearsonr(s1_aligned, s2_aligned)

                    correlation_data[symbol1][symbol2] = corr if not np.isnan(corr) else 0.0
                except Exception as e:
                    correlation_data[symbol1][symbol2] = 0.0

        self.correlation_matrix = pd.DataFrame(correlation_data)
        self.last_matrix_update = now

        # Зберігаємо матрицю в базі даних
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Перетворюємо DataFrame на JSON
            matrix_json = self.correlation_matrix.to_json()

            # Зберігаємо в базі
            cursor.execute('''
            INSERT INTO correlation_matrix (timestamp, matrix_data)
            VALUES (?, ?)
            ''', (now.isoformat(), matrix_json))

            # Видаляємо старі матриці, залишаємо тільки 5 останніх
            cursor.execute('''
            DELETE FROM correlation_matrix 
            WHERE id NOT IN (
                SELECT id FROM correlation_matrix 
                ORDER BY timestamp DESC 
                LIMIT 5
            )
            ''')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving correlation matrix: {e}")

    async def analyze_market_correlation(self, symbol, all_symbols):
        """
        Аналіз кореляції з іншими маркетами для виявлення скоординованих pump-and-dump атак

        :param symbol: Поточний символ для аналізу
        :param all_symbols: Список всіх символів для порівняння
        :return: Dict з результатами аналізу
        """
        # Отримуємо базову валюту (наприклад, 'BTC' з 'BTC/USDT')
        base_currency = symbol.split('/')[0]

        # Ми аналізуємо тільки альткоїни (не BTC, ETH, BNB тощо)
        major_coins = {'BTC', 'ETH', 'BNB', 'USDT', 'USDC', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE'}
        if base_currency in major_coins:
            return {'correlation_signal': False}

        # Аналізуємо дані для поточного символу
        df = self._get_dataframe_for_symbol(symbol, hours=3)  # Аналізуємо останні 3 години

        if df is None or len(df) < 5:
            return {'correlation_signal': False}

        # Виявляємо різке зростання ціни (pump)
        last_hour_data = df.iloc[-12:]  # Останні 12 5-хвилинних інтервалів = 1 година
        if len(last_hour_data) < 2:
            return {'correlation_signal': False}

        price_change = (last_hour_data['price'].iloc[-1] / last_hour_data['price'].iloc[0] - 1) * 100

        # Оновлюємо кореляційну матрицю, якщо це необхідно
        self._update_correlation_matrix()

        # Перевіряємо чи виконуються умови для pump-and-dump сигналу
        pump_signal = price_change >= self.pump_threshold

        # Якщо немає різкого зростання або немає матриці кореляції, завершуємо аналіз
        if not pump_signal or self.correlation_matrix is None or symbol not in self.correlation_matrix:
            return {'correlation_signal': False}

        # Знаходимо монети, які корелюють з даною монетою
        correlated_coins = []
        highly_correlated = None

        try:
            correlations = self.correlation_matrix[symbol].dropna()
            highly_correlated = correlations[correlations >= self.correlation_threshold]

            # Виключаємо саму монету з результатів
            highly_correlated = highly_correlated[highly_correlated.index != symbol]

            correlated_coins = highly_correlated.index.tolist()

            # Додатково перевіряємо, чи ростуть ці монети також
            confirmed_correlated = []
            for coin in correlated_coins:
                coin_df = self._get_dataframe_for_symbol(coin, hours=3)
                if coin_df is not None and len(coin_df) >= 5:
                    coin_last_hour = coin_df.iloc[-12:]
                    if len(coin_last_hour) < 2:
                        continue

                    coin_price_change = (coin_last_hour['price'].iloc[-1] /
                                         coin_last_hour['price'].iloc[0] - 1) * 100

                    if coin_price_change >= self.pump_threshold * 0.7:  # Дозволяємо невелике відхилення
                        confirmed_correlated.append(coin)

            correlated_coins = confirmed_correlated
        except Exception as e:
            # Якщо виникла помилка, просто повертаємо порожній список
            correlated_coins = []

        # Визначаємо тип кореляції
        correlation_type = 'normal'
        if pump_signal and len(correlated_coins) >= 2:
            correlation_type = 'pump_group'
        elif pump_signal:
            correlation_type = 'single_pump'

        correlation_signal = pump_signal and len(correlated_coins) >= 2

        correlation_strength = 0.0  # Значення за замовчуванням
        if correlated_coins and highly_correlated is not None:
            # Беремо середнє значення кореляцій для знайдених монет
            correlation_values = [highly_correlated[coin] for coin in correlated_coins if coin in highly_correlated]
            if correlation_values:
                correlation_strength = sum(correlation_values) / len(correlation_values)

        return {
            'correlation_signal': correlation_signal,
            'correlated_coins': correlated_coins,
            'correlation_type': correlation_type,
            'price_change_1h': price_change,
            'correlation_strength': correlation_strength  # Додаємо цей параметр
        }

    def add_market_data(self, symbol, price, volume, timestamp=None):
        """
        Додавання даних про ринкову активність для подальшого аналізу

        :param symbol: Символ криптовалюти
        :param price: Ціна
        :param volume: Об'єм торгів
        :param timestamp: Часова мітка (за замовчуванням - поточний час)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Переконуємося, що ціна та обсяг є числовими значеннями
        try:
            price = float(price)
            volume = float(volume)
        except (ValueError, TypeError):
            return

        # Зберігаємо дані в базі
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
            INSERT OR REPLACE INTO market_data (symbol, timestamp, price, volume)
            VALUES (?, ?, ?, ?)
            ''', (symbol, timestamp.isoformat(), price, volume))

            # Видаляємо старі дані для економії місця
            cutoff_time = datetime.now() - timedelta(hours=self.window_size * 1.5)
            cursor.execute('''
            DELETE FROM market_data 
            WHERE symbol = ? AND timestamp < ?
            ''', (symbol, cutoff_time.isoformat()))

            conn.commit()
        except Exception as e:
            print(f"Error adding market data for {symbol}: {e}")
        finally:
            conn.close()

        # Invalidate cache for this symbol
        for key in list(self.cache_expiry.keys()):
            if key.startswith(symbol + "_"):
                if key in self.market_data_cache:
                    del self.market_data_cache[key]
                if key in self.cache_expiry:
                    del self.cache_expiry[key]

        # Сигналізуємо про необхідність оновлення матриці кореляцій
        self.last_matrix_update = None

    def get_correlated_groups(self, threshold=None):
        """
        Отримання груп корельованих активів

        :param threshold: Поріг кореляції для об'єднання в групу (якщо None, використовується self.correlation_threshold)
        :return: Список груп корельованих активів
        """
        if threshold is None:
            threshold = self.correlation_threshold

        # Оновлюємо кореляційну матрицю
        self._update_correlation_matrix()

        if self.correlation_matrix is None:
            return []

        # Перетворюємо матрицю кореляції у формат, придатний для кластеризації
        matrix = self.correlation_matrix.values
        np.fill_diagonal(matrix, 0)  # Видаляємо діагональ

        # Перетворюємо кореляцію у відстань (1 - кореляція)
        distance_matrix = 1 - np.abs(matrix)

        # Використовуємо DBSCAN для кластеризації
        try:
            clustering = DBSCAN(eps=1 - threshold, min_samples=2, metric='precomputed').fit(distance_matrix)

            # Отримуємо мітки кластерів для кожного символу
            labels = clustering.labels_
            symbols = list(self.correlation_matrix.columns)

            # Формуємо групи символів
            groups = {}
            for i, label in enumerate(labels):
                if label == -1:  # -1 означає, що точка не належить жодному кластеру
                    continue

                if label not in groups:
                    groups[label] = []

                groups[label].append(symbols[i])

            return list(groups.values())
        except Exception as e:
            # Якщо виникла помилка з DBSCAN, використовуємо простіший підхід
            return self._get_groups_manually(threshold)

    def _get_groups_manually(self, threshold):
        """
        Резервний метод для отримання груп корельованих активів

        :param threshold: Поріг кореляції
        :return: Список груп корельованих активів
        """
        if self.correlation_matrix is None:
            return []

        groups = []
        processed = set()

        for symbol in self.correlation_matrix.columns:
            if symbol in processed:
                continue

            # Знаходимо всі монети, які корелюють з поточною
            correlated = self.correlation_matrix[symbol][
                self.correlation_matrix[symbol] >= threshold
                ].index.tolist()

            # Виключаємо вже оброблені монети та саму монету
            correlated = [s for s in correlated if s != symbol and s not in processed]

            if correlated:
                group = [symbol] + correlated
                groups.append(group)
                processed.update(group)

        return groups

    def get_price_change(self, symbol, hours=1):
        """
        Отримання зміни ціни за вказаний період

        :param symbol: Символ криптовалюти
        :param hours: Кількість годин для аналізу
        :return: Відсоткова зміна ціни за вказаний період або None, якщо недостатньо даних
        """
        df = self._get_dataframe_for_symbol(symbol, hours=hours)

        if df is None or len(df) < 2:
            return None

        price_start = df['price'].iloc[0]
        price_end = df['price'].iloc[-1]

        return (price_end / price_start - 1) * 100

    def clear_cache(self):
        """
        Очистити кеш даних
        """
        self.market_data_cache.clear()
        self.cache_expiry.clear()
        self.last_matrix_update = None

    def get_database_stats(self):
        """
        Отримання статистики бази даних

        :return: Словник зі статистикою
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Підрахунок загальної кількості записів
        cursor.execute("SELECT COUNT(*) FROM market_data")
        total_records = cursor.fetchone()[0]

        # Кількість унікальних символів
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
        unique_symbols = cursor.fetchone()[0]

        # Розмір файлу бази даних
        db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # в МБ

        # Найновіші дані
        cursor.execute("SELECT MAX(timestamp) FROM market_data")
        latest_data = cursor.fetchone()[0]

        # Найстаріші дані
        cursor.execute("SELECT MIN(timestamp) FROM market_data")
        oldest_data = cursor.fetchone()[0]

        conn.close()

        return {
            "total_records": total_records,
            "unique_symbols": unique_symbols,
            "database_size_mb": db_size,
            "latest_data": latest_data,
            "oldest_data": oldest_data
        }