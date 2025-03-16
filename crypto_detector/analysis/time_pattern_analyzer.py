from datetime import datetime


class TimePatternAnalyzer:
    """
    Клас для аналізу часових патернів, пов'язаних з підозрілою активністю.
    Певні години доби та дні тижня мають статистично підвищений ризик pump-and-dump схем.
    """

    def __init__(self):
        """
        Ініціалізація аналізатора часових патернів
        """
        # Години підвищеного ризику (ніч та пізній вечір)
        self.high_risk_hours = {0, 1, 2, 3, 4, 20, 21, 22, 23}

    async def check_time_pattern(self):
        """
        Перевірка чи поточний час відповідає часу, коли часто відбуваються pump-and-dump схеми

        :return: Dict з інформацією про часові патерни
        """
        current_time = datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()  # 0 = Monday, 6 = Sunday

        # Години коли частіше відбуваються памп схеми
        is_high_risk_hour = current_hour in self.high_risk_hours

        # Вихідні часто є більш ризиковими
        is_weekend = current_weekday >= 5

        # Загальна оцінка ризику часу
        time_risk = 0.0
        if is_high_risk_hour:
            time_risk += 0.7
        if is_weekend:
            time_risk += 0.3

        return {
            'time_pattern_signal': time_risk > 0.5,
            'is_high_risk_hour': is_high_risk_hour,
            'is_weekend': is_weekend,
            'time_risk_score': time_risk
        }

    def set_high_risk_hours(self, hours_set):
        """
        Встановлення нових годин підвищеного ризику

        :param hours_set: Множина годин (0-23)
        """
        self.high_risk_hours = set(hours_set)
