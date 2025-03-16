import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

from crypto_detector.config.settings import VISUALIZATION_COLORS


class DetectorVisualizer:
    """
    Клас для візуалізації результатів роботи детектора підозрілої активності.
    Дозволяє наочно представити дані, сигнали та аномалії.
    """

    def __init__(self):
        """
        Ініціалізація візуалізатора
        """
        # Налаштування стилю
        self.colors = VISUALIZATION_COLORS
        sns.set_style("whitegrid")

    def plot_detector_results(self, historical_data, detection_results, threshold=0.5, title=None):
        """
        Візуалізація результатів виявлення підозрілої активності

        :param historical_data: DataFrame з історичними OHLCV даними
        :param detection_results: Список результатів виявлення (кожен результат - словник)
        :param threshold: Поріг для класифікації
        :param title: Заголовок графіка
        """
        if not detection_results:
            print("Немає результатів для візуалізації")
            return

        # Перетворення результатів у DataFrame
        results_df = pd.DataFrame([
            {
                'timestamp': datetime.fromisoformat(r['timestamp']),
                'probability': r['probability_score'],
                'signals_count': len(r['signals'])
            } for r in detection_results
        ])
        results_df.set_index('timestamp', inplace=True)
        results_df.sort_index(inplace=True)

        # Визначення часового діапазону
        start_time = historical_data.index[0]
        end_time = historical_data.index[-1]

        # Створення графіка
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Графік ціни
        ax1.plot(historical_data.index, historical_data['close'], color=self.colors['price'], label='Ціна закриття')

        # Виділення точок виявлення
        for timestamp, row in results_df.iterrows():
            if timestamp >= start_time and timestamp <= end_time:
                color = 'red' if row['probability'] > threshold else 'blue'
                alpha = 0.3 + min(row['probability'] * 0.7, 0.7)  # Прозорість залежно від ймовірності
                ax1.axvline(x=timestamp, color=color, alpha=alpha, linewidth=1)

        ax1.set_title(title or 'Результати виявлення підозрілої активності')
        ax1.set_ylabel('Ціна')
        ax1.legend()
        ax1.grid(True)

        # Графік об'єму
        ax2.bar(historical_data.index, historical_data['volume'], color=self.colors['volume'], alpha=0.7,
                label='Об\'єм')
        ax2.set_ylabel('Об\'єм')
        ax2.legend()
        ax2.grid(True)

        # Графік ймовірності підозрілої активності
        ax3.plot(results_df.index, results_df['probability'], color=self.colors['prediction'], marker='o',
                 linestyle='-', label='Ймовірність підозрілої активності')
        ax3.axhline(y=threshold, color=self.colors['threshold'], linestyle='--', alpha=0.7,
                    label=f'Поріг ({threshold})')

        # Заливка області вище порогу
        ax3.fill_between(results_df.index, results_df['probability'], threshold,
                         where=(results_df['probability'] > threshold),
                         color='red', alpha=0.3, interpolate=True)

        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel('Ймовірність')
        ax3.set_xlabel('Дата')
        ax3.legend()
        ax3.grid(True)

        # Форматування осі X
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_signal_contributions(self, detection_result, title=None):
        """
        Візуалізація внеску кожного сигналу

        :param detection_result: Результат виявлення підозрілої активності
        :param title: Заголовок графіка
        """
        if not detection_result or 'signals' not in detection_result:
            print("Немає сигналів для візуалізації")
            return

        # Збір даних про сигнали
        signals = detection_result['signals']
        total_probability = detection_result['probability_score']

        # Створення DataFrame для візуалізації
        signals_df = pd.DataFrame([
            {
                'signal': s['name'],
                'weight': s['weight'],
                'contribution': s['weight'] * total_probability / (1.0 if not s.get('weight') else s['weight'])
            } for s in signals
        ])

        if signals_df.empty:
            print("Немає сигналів для візуалізації")
            return

        # Сортування за внеском
        signals_df.sort_values('contribution', ascending=False, inplace=True)

        # Створення графіка
        plt.figure(figsize=(10, 6))
        bars = plt.barh(signals_df['signal'], signals_df['contribution'], color='skyblue')

        # Додавання значень
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{signals_df['contribution'].iloc[i]:.3f}",
                     va='center')

        # Налаштування графіка
        plt.title(title or f'Внесок сигналів (Загальна ймовірність: {total_probability:.4f})')
        plt.xlabel('Внесок')
        plt.ylabel('Сигнал')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlim(0, max(signals_df['contribution']) * 1.2)
        plt.tight_layout()
        plt.show()

    def plot_volume_analysis(self, ohlcv_data, volume_analysis, title=None):
        """
        Візуалізація аналізу об'єму

        :param ohlcv_data: DataFrame з OHLCV даними
        :param volume_analysis: Результат аналізу об'єму
        :param title: Заголовок графіка
        """
        if ohlcv_data.empty or not volume_analysis:
            print("Недостатньо даних для візуалізації аналізу об'єму")
            return

        # Створення графіка
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Графік об'єму
        ax1.bar(ohlcv_data.index, ohlcv_data['volume'], color=self.colors['volume'], alpha=0.7, label='Об\'єм')
        ax1.axhline(y=volume_analysis.get('mean_volume', 0), color='red', linestyle='--',
                    label=f'Середній об\'єм: {volume_analysis.get("mean_volume", 0):.2f}')
        ax1.axhline(y=volume_analysis.get('recent_volume', 0), color='green', linestyle='--',
                    label=f'Останній об\'єм: {volume_analysis.get("recent_volume", 0):.2f}')

        # Виділення останніх 5 свічок
        recent_data = ohlcv_data.iloc[-5:]
        ax1.bar(recent_data.index, recent_data['volume'], color='red', alpha=0.7, label='Останні дані')

        ax1.set_title(title or 'Аналіз об\'єму')
        ax1.set_ylabel('Об\'єм')
        ax1.legend()
        ax1.grid(True)

        # Графік зміни об'єму
        if 'volume_change' not in ohlcv_data.columns:
            ohlcv_data['volume_change'] = ohlcv_data['volume'].pct_change() * 100

        ax2.plot(ohlcv_data.index, ohlcv_data['volume_change'], color='blue', label='Зміна об\'єму (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Виділення останніх 5 свічок
        ax2.plot(recent_data.index, recent_data['volume_change'], color='red', linewidth=2)

        # Додавання z-score, якщо доступний
        if 'z_score' in volume_analysis:
            z_score = volume_analysis['z_score']
            ax2.axhline(y=z_score, color='purple', linestyle='--',
                        label=f'Z-score: {z_score:.2f}')

        ax2.set_ylabel('Зміна об\'єму (%)')
        ax2.set_xlabel('Дата')
        ax2.legend()
        ax2.grid(True)

        # Форматування осі X
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Додавання анотацій
        plt.figtext(0.01, 0.01,
                    f"Аномальний об'єм: {volume_analysis.get('unusual_volume', False)}\n" +
                    f"Зміна об'єму: {volume_analysis.get('percent_change', 0):.2f}%\n" +
                    f"Z-score: {volume_analysis.get('z_score', 0):.2f}",
                    fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_price_analysis(self, ohlcv_data, price_analysis, title=None):
        """
        Візуалізація аналізу цінової динаміки

        :param ohlcv_data: DataFrame з OHLCV даними
        :param price_analysis: Результат аналізу цінової динаміки
        :param title: Заголовок графіка
        """
        if ohlcv_data.empty or not price_analysis:
            print("Недостатньо даних для візуалізації аналізу цінової динаміки")
            return

        # Створення графіка
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Графік ціни
        ax1.plot(ohlcv_data.index, ohlcv_data['close'], color=self.colors['price'], label='Ціна закриття')

        # Виділення останніх 5 свічок
        recent_data = ohlcv_data.iloc[-5:]
        ax1.plot(recent_data.index, recent_data['close'], color='red', linewidth=2, label='Останні дані')

        ax1.set_title(title or 'Аналіз цінової динаміки')
        ax1.set_ylabel('Ціна')
        ax1.legend()
        ax1.grid(True)

        # Графік прибутковості
        if 'returns' not in ohlcv_data.columns:
            ohlcv_data['returns'] = ohlcv_data['close'].pct_change() * 100

        ax2.plot(ohlcv_data.index, ohlcv_data['returns'], color='blue', label='Прибутковість (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Виділення останніх 5 свічок
        ax2.plot(recent_data.index, recent_data['returns'], color='red', linewidth=2)

        ax2.set_ylabel('Прибутковість (%)')
        ax2.set_xlabel('Дата')
        ax2.legend()
        ax2.grid(True)

        # Форматування осі X
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Додавання анотацій
        price_change_1h = price_analysis.get('price_change_1h', 0)
        price_change_24h = price_analysis.get('price_change_24h', 0)
        volatility_ratio = price_analysis.get('volatility_ratio', 0)
        consecutive_up = price_analysis.get('consecutive_up', 0)

        plt.figtext(0.01, 0.01,
                    f"Зміна ціни (1 год): {price_change_1h:.2f}%\n" +
                    f"Зміна ціни (24 год): {price_change_24h:.2f}%\n" +
                    f"Співвідношення волатильності: {volatility_ratio:.2f}\n" +
                    f"Послідовних зростань: {consecutive_up}/5",
                    fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_orderbook_analysis(self, order_book, order_book_analysis, title=None):
        """
        Візуалізація аналізу книги ордерів

        :param order_book: Дані книги ордерів
        :param order_book_analysis: Результат аналізу книги ордерів
        :param title: Заголовок графіка
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            print("Недостатньо даних для візуалізації аналізу книги ордерів")
            return

        # Підготовка даних
        bids = order_book['bids'][:20]  # Обмеження до 20 рівнів
        asks = order_book['asks'][:20]

        bid_prices = [bid[0] for bid in bids]
        bid_volumes = [bid[1] for bid in bids]
        ask_prices = [ask[0] for ask in asks]
        ask_volumes = [ask[1] for ask in asks]

        # Створення графіка
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Графік цін
        ax1.barh(range(len(bid_prices)), bid_volumes, height=0.4, color='green', alpha=0.6, label='Покупка (Bid)')
        ax1.barh([i + 0.5 for i in range(len(ask_prices))], ask_volumes, height=0.4, color='red', alpha=0.6,
                 label='Продаж (Ask)')

        # Додавання цін до міток
        ax1.set_yticks([i + 0.25 for i in range(max(len(bid_prices), len(ask_prices)))])

        combined_prices = []
        for i in range(max(len(bid_prices), len(ask_prices))):
            bid_price = bid_prices[i] if i < len(bid_prices) else ""
            ask_price = ask_prices[i] if i < len(ask_prices) else ""
            combined_prices.append(f"{bid_price} | {ask_price}")

        ax1.set_yticklabels(combined_prices)
        ax1.set_title('Книга ордерів: Ціни та Об\'єми')
        ax1.set_xlabel('Об\'єм')
        ax1.legend()
        ax1.grid(True, axis='x')

        # Графік кумулятивного об'єму
        cumulative_bids = np.cumsum(bid_volumes)
        cumulative_asks = np.cumsum(ask_volumes)

        ax2.step(bid_prices, cumulative_bids, where='post', color='green', alpha=0.6, label='Кумулятивно Bid')
        ax2.step(ask_prices, cumulative_asks, where='post', color='red', alpha=0.6, label='Кумулятивно Ask')
        ax2.fill_between(bid_prices, cumulative_bids, step='post', alpha=0.1, color='green')
        ax2.fill_between(ask_prices, cumulative_asks, step='post', alpha=0.1, color='red')

        ax2.set_title('Кумулятивний об\'єм')
        ax2.set_xlabel('Ціна')
        ax2.set_ylabel('Кумулятивний об\'єм')
        ax2.legend()
        ax2.grid(True)

        # Додавання анотацій
        buy_sell_ratio = order_book_analysis.get('buy_sell_ratio', 1.0)
        buy_volume = order_book_analysis.get('buy_volume', 0)
        sell_volume = order_book_analysis.get('sell_volume', 0)
        signal = order_book_analysis.get('order_book_signal', False)

        plt.figtext(0.4, 0.02,
                    f"Співвідношення покупка/продаж: {buy_sell_ratio:.2f}\n" +
                    f"Об\'єм покупок: {buy_volume:.2f}\n" +
                    f"Об\'єм продажів: {sell_volume:.2f}\n" +
                    f"Сигнал дисбалансу: {signal}",
                    fontsize=10)

        plt.tight_layout()
        plt.suptitle(title or 'Аналіз книги ордерів', y=1.02)
        plt.show()

    def plot_social_analysis(self, social_data, timepoints=None, title=None):
        """
        Візуалізація аналізу соціальної активності

        :param social_data: Історія соціальної активності
        :param timepoints: Часові мітки для відображення (якщо None, використовуються всі)
        :param title: Заголовок графіка
        """
        if not social_data or not isinstance(social_data, dict):
            print("Недостатньо даних для візуалізації аналізу соціальної активності")
            return

        # Створення часових міток, якщо не надані
        if timepoints is None:
            now = datetime.now()
            timepoints = [now - timedelta(hours=i) for i in range(len(next(iter(social_data.values()))))]

        # Створення DataFrame для візуалізації
        social_df = pd.DataFrame()

        for coin, mentions in social_data.items():
            if len(mentions) == len(timepoints):
                social_df[coin] = mentions

        if social_df.empty:
            print("Недостатньо даних для візуалізації аналізу соціальної активності")
            return

        social_df.index = timepoints
        social_df.sort_index(inplace=True)

        # Створення графіка
        plt.figure(figsize=(12, 6))

        for coin in social_df.columns:
            plt.plot(social_df.index, social_df[coin], marker='o', label=coin)

        plt.title(title or 'Аналіз соціальної активності')
        plt.ylabel('Кількість згадок')
        plt.xlabel('Дата/Час')
        plt.legend()
        plt.grid(True)

        # Форматування осі X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_detected_events(self, historical_data, events_df, title=None):
        """
        Візуалізація виявлених подій

        :param historical_data: DataFrame з історичними OHLCV даними
        :param events_df: DataFrame з подіями
        :param title: Заголовок графіка
        """
        if events_df.empty:
            print("Немає подій для візуалізації")
            return

        # Створення графіка
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Графік ціни
        ax1.plot(historical_data.index, historical_data['close'], color=self.colors['price'], alpha=0.7,
                 label='Ціна закриття')

        # Позначення подій
        for idx, event in events_df.iterrows():
            if idx in historical_data.index:
                event_type = event.get('event_type', 'unknown')
                color = 'red' if event_type == 'pump_and_dump' else 'orange'
                ax1.axvline(x=idx, color=color, alpha=0.7, linestyle='--')
                ax1.plot(idx, historical_data.loc[idx, 'close'], 'o', color=color, markersize=8)

                # Додавання анотації
                ax1.annotate(f"{event_type}",
                             xy=(idx, historical_data.loc[idx, 'close']),
                             xytext=(10, 20),
                             textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

        ax1.set_title(title or 'Виявлені події')
        ax1.set_ylabel('Ціна')
        ax1.legend()
        ax1.grid(True)

        # Графік об'єму
        ax2.bar(historical_data.index, historical_data['volume'], color=self.colors['volume'], alpha=0.7,
                label='Об\'єм')

        # Позначення подій
        for idx, event in events_df.iterrows():
            if idx in historical_data.index:
                event_type = event.get('event_type', 'unknown')
                color = 'red' if event_type == 'pump_and_dump' else 'orange'
                ax2.axvline(x=idx, color=color, alpha=0.7, linestyle='--')

        ax2.set_ylabel('Об\'єм')
        ax2.set_xlabel('Дата')
        ax2.legend()
        ax2.grid(True)

        # Форматування осі X
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

        # Додатковий графік для деталей подій
        if 'pump_percent' in events_df.columns and 'dump_percent' in events_df.columns:
            plt.figure(figsize=(10, 6))

            events_to_plot = events_df[['pump_percent', 'dump_percent', 'event_type']].copy()
            events_to_plot['abs_dump'] = events_to_plot['dump_percent'].abs()

            # Графік для порівняння pump і dump
            ax = events_to_plot.plot(kind='bar', y=['pump_percent', 'abs_dump'], color=['green', 'red'], alpha=0.7)

            plt.title('Порівняння Pump і Dump фаз для кожної події')
            plt.ylabel('Відсоток зміни (%)')
            plt.xlabel('Дата події')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.legend(['Pump (%)', 'Dump (%)'])
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()
