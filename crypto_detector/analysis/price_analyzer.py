import pandas as pd
import numpy as np


class PriceAnalyzer:
    """
    Клас для аналізу цінової динаміки криптовалют.
    Виявляє незвичайні рухи ціни та паттерни, які можуть вказувати на pump-and-dump схеми.
    """

    def __init__(self):
        """
        Ініціалізація аналізатора цінової динаміки
        """
        pass

    async def detect_price_action(self, df_ohlcv):
        """
        Аналіз цінової динаміки з покращеним виявленням pump-and-dump схем

        :param df_ohlcv: DataFrame з OHLCV даними
        :return: Dict з результатами аналізу
        """
        if df_ohlcv.empty or len(df_ohlcv) < 10:
            return {'price_action_signal': False}

        # Розрахунок прибутковості
        df_ohlcv['returns'] = df_ohlcv['close'].pct_change()

        # Розрахунок волатильності (стандартне відхилення прибутковості)
        historical_volatility = df_ohlcv['returns'][:-5].std()
        recent_volatility = df_ohlcv['returns'][-5:].std()

        # Розрахунок зміни ціни для різних часових діапазонів
        price_change_1h = (df_ohlcv['close'].iloc[-1] / df_ohlcv['close'].iloc[-12] - 1) * 100 if len(
            df_ohlcv) >= 12 else 0
        price_change_24h = (df_ohlcv['close'].iloc[-1] / df_ohlcv['close'].iloc[0] - 1) * 100 if len(
            df_ohlcv) > 1 else 0

        # Новий розрахунок: відстань до локального максимуму
        window_high = df_ohlcv['high'].max()
        current_price = df_ohlcv['close'].iloc[-1]
        distance_from_high = (current_price / window_high - 1) * 100

        # Визначення, чи був пік ціни в останні 48 годин
        high_idx = df_ohlcv['high'].idxmax()
        latest_idx = df_ohlcv.index[-1]
        try:
            hours_since_peak = abs((latest_idx - high_idx).total_seconds()) / 3600
            recent_peak = hours_since_peak < 48
        except:
            recent_peak = False

        # Пошук різких рухів ціни (свічки з великими тілами)
        df_ohlcv['body_size'] = abs(df_ohlcv['close'] - df_ohlcv['open']) / df_ohlcv['open'] * 100
        large_candles = (df_ohlcv['body_size'][-5:] > 2.5).sum()

        # Виявлення патернів підвищення ціни
        df_ohlcv['price_change'] = df_ohlcv['close'].pct_change()
        consecutive_up = sum(1 for x in df_ohlcv['price_change'][-5:] if x > 0)

        # Виявлення прискорення руху ціни
        price_acceleration = df_ohlcv['price_change'].diff()[-5:].mean() if len(df_ohlcv) >= 6 else 0

        # Розрахунок експоненційного скользящого середнього для виявлення тренду
        if len(df_ohlcv) >= 20:
            df_ohlcv['ema20'] = df_ohlcv['close'].ewm(span=20).mean()
            price_above_ema = df_ohlcv['close'].iloc[-1] > df_ohlcv['ema20'].iloc[-1]
        else:
            price_above_ema = False

        # Поліпшена логіка сигналу для виявлення pump-and-dump
        pump_signal = price_change_1h > 3 or recent_volatility > historical_volatility * 1.5 or consecutive_up >= 4 or large_candles >= 2

        # Додаткова перевірка на значну зміну ціни за 24 години (pump фаза)
        significant_pump = price_change_24h > 50

        # Ознака dump фази (ціна опустилася значно нижче пікової)
        dump_phase = distance_from_high < -15 and recent_peak

        # Комбінований сигнал
        price_action_signal = pump_signal or significant_pump or dump_phase

        return {
            'price_action_signal': price_action_signal,
            'price_change_1h': price_change_1h,
            'price_change_24h': price_change_24h,
            'volatility_ratio': recent_volatility / historical_volatility if historical_volatility > 0 else 0,
            'large_candles': large_candles,
            'consecutive_up': consecutive_up,
            'price_acceleration': price_acceleration,
            'distance_from_high': distance_from_high,
            'hours_since_peak': hours_since_peak if recent_peak else None,
            'significant_pump': significant_pump,
            'dump_phase': dump_phase,
            'price_above_ema': price_above_ema
        }

    async def analyze_historical_price(self, data_window):
        """
        Аналіз історичної цінової динаміки з додатковими показниками

        :param data_window: Вікно даних для аналізу
        :return: Результати аналізу ціни
        """
        if data_window.empty or len(data_window) < 10:
            return {'price_action_signal': False, 'recent_price_change': 0}

        # Розрахунок прибутковості
        data_window['returns'] = data_window['close'].pct_change()

        # Розрахунок волатильності (стандартне відхилення прибутковості)
        historical_volatility = data_window['returns'][:-5].std()
        recent_volatility = data_window['returns'][-5:].std()

        # Розрахунок зміни ціни
        recent_price_change = (data_window['close'].iloc[-1] / data_window['close'].iloc[-6] - 1) * 100

        # Розрахунок швидкості зміни ціни
        price_velocity = data_window['returns'][-5:].mean() * 100

        # Виявлення послідовних зростань
        consecutive_up = sum(1 for x in data_window['returns'][-5:] if x > 0)

        # Пошук свічок з великими тілами
        data_window['body_size'] = abs(data_window['close'] - data_window['open']) / data_window['open'] * 100
        large_candles = (data_window['body_size'][-5:] > 2.5).sum()

        return {
            'price_action_signal': (recent_price_change > 3 or  # Знижено поріг для чутливості
                                    recent_volatility > historical_volatility * 1.5 or
                                    consecutive_up >= 4 or
                                    large_candles >= 2),
            'recent_price_change': recent_price_change,
            'volatility_ratio': recent_volatility / historical_volatility if historical_volatility > 0 else 0,
            'price_velocity': price_velocity,
            'consecutive_up': consecutive_up,
            'large_candles': large_candles
        }
