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

        vertical_jump, jump_percent = self.detect_vertical_price_jump(df_ohlcv)
        v_pattern_detected = self.detect_v_pattern(df_ohlcv)
        large_green_candle, candle_body_percent = self.detect_large_candles(df_ohlcv)

        return {
            'price_action_signal': price_action_signal or vertical_jump or v_pattern_detected or large_green_candle,
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
            'price_above_ema': price_above_ema,
            'vertical_price_jump': vertical_jump,
            'jump_percent': jump_percent,
            'v_pattern_detected': v_pattern_detected,
            'large_green_candle': large_green_candle,
            'candle_body_percent': candle_body_percent
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

    def detect_vertical_price_jump(self, df_ohlcv, threshold_percent=15, window=5):
        """Виявляє вертикальний стрибок ціни"""
        if len(df_ohlcv) < window + 1:
            return False, 0

        # Розрахунок процентної зміни за останнє вікно
        current_price = df_ohlcv['close'].iloc[-1]
        start_price = df_ohlcv['close'].iloc[-window - 1] if len(df_ohlcv) > window + 1 else df_ohlcv['close'].iloc[0]
        jump_percent = ((current_price / start_price) - 1) * 100

        # Перевірка об'єму
        if len(df_ohlcv) > window:
            avg_volume = df_ohlcv['volume'][:-window].mean()
            recent_volume = df_ohlcv['volume'][-window:].mean()
            volume_increase = recent_volume > avg_volume * 2
        else:
            volume_increase = False

        return jump_percent > threshold_percent and volume_increase, jump_percent

    def detect_v_pattern(self, df_ohlcv, rise_threshold=20, fall_threshold=10, max_periods=24):
        """Виявляє V-подібний патерн (швидке зростання і падіння)"""
        if len(df_ohlcv) < max_periods:
            return False

        # Обмежуємо аналіз останніми max_periods свічками
        window = df_ohlcv.iloc[-min(len(df_ohlcv), max_periods):]

        # Знаходимо максимальну ціну у вікні
        high_price = window['high'].max()

        # Знаходимо ціну на початку вікна та поточну ціну
        start_price = window['close'].iloc[0]
        current_price = window['close'].iloc[-1]

        # Розрахунок процентних змін
        rise_percent = ((high_price / start_price) - 1) * 100
        fall_percent = ((current_price / high_price) - 1) * 100

        return rise_percent > rise_threshold and fall_percent < -fall_threshold

    def detect_large_candles(self, df_ohlcv, body_threshold=3.0, window=5):
        """Виявляє свічки з великим тілом"""
        if len(df_ohlcv) < 2:
            return False, 0

        # Створюємо копію для безпечних змін
        df_temp = df_ohlcv.copy()

        # Розрахунок розміру тіла свічок
        df_temp['body_size'] = abs(df_temp['close'] - df_temp['open']) / df_temp['open'] * 100
        df_temp['is_green'] = df_temp['close'] > df_temp['open']

        # Виявлення великих зелених свічок у останньому вікні
        window_size = min(len(df_temp), window)
        large_green_candles = df_temp[(df_temp['body_size'] > body_threshold) &
                                      (df_temp['is_green'])].iloc[-window_size:]

        if not large_green_candles.empty:
            max_body = large_green_candles['body_size'].max()
            return True, max_body

        return False, 0
