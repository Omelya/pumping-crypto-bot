import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from crypto_detector.config.settings import VISUALIZATION_COLORS


class BacktestVisualizer:
    """
    Клас для візуалізації результатів бектестингу.
    Дозволяє наочно представити результати та метрики бектестингу.
    """

    def __init__(self):
        """
        Ініціалізація візуалізатора
        """
        # Налаштування стилю
        self.colors = VISUALIZATION_COLORS
        sns.set_style("whitegrid")

    def plot_backtest_results(self, predictions_df, symbol, threshold=None, title=None):
        """
        Візуалізація результатів бектестингу

        :param predictions_df: DataFrame з результатами бектестингу
        :param symbol: Символ криптовалюти
        :param threshold: Поріг для класифікації (якщо None, використовується 0.5)
        :param title: Заголовок графіка
        """
        if predictions_df.empty:
            print("Немає даних для візуалізації")
            return

        threshold = threshold or 0.5

        # Конвертація часових міток, якщо вони зберігаються як рядки
        if 'timestamp' in predictions_df.columns and not pd.api.types.is_datetime64_any_dtype(
                predictions_df['timestamp']):
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            predictions_df.set_index('timestamp', inplace=True)

        # Створення графіка
        plt.figure(figsize=(12, 6))

        # Розділення на події та не-події
        events = predictions_df[predictions_df['is_event'] == 1]
        non_events = predictions_df[predictions_df['is_event'] == 0]

        # Графік ймовірностей
        plt.scatter(non_events.index, non_events['probability'],
                    color='blue', alpha=0.6, label='Не-події')
        plt.scatter(events.index, events['probability'],
                    color='red', alpha=0.6, label='Події')

        # Лінія порогу
        plt.axhline(y=threshold, color=self.colors['threshold'], linestyle='--', alpha=0.7,
                    label=f'Поріг ({threshold})')

        # Додавання тренду
        if len(predictions_df) > 1:
            z = np.polyfit(range(len(predictions_df)), predictions_df['probability'], 1)
            p = np.poly1d(z)
            plt.plot(predictions_df.index, p(range(len(predictions_df))),
                     color='purple', linestyle='-', alpha=0.5, label='Тренд')

        # Виділення правильних та неправильних передбачень
        true_positives = events[events['prediction'] == 1]
        false_negatives = events[events['prediction'] == 0]
        true_negatives = non_events[non_events['prediction'] == 0]
        false_positives = non_events[non_events['prediction'] == 1]

        plt.scatter(true_positives.index, true_positives['probability'],
                    color='green', marker='o', s=100, alpha=0.7, label='True Positive')
        plt.scatter(false_negatives.index, false_negatives['probability'],
                    color='purple', marker='X', s=100, alpha=0.7, label='False Negative')
        plt.scatter(false_positives.index, false_positives['probability'],
                    color='orange', marker='X', s=100, alpha=0.7, label='False Positive')

        # Налаштування графіка
        plt.title(title or f'Результати бектестингу для {symbol}')
        plt.ylabel('Ймовірність')
        plt.xlabel('Дата')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)

        # Форматування осі X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_metrics_comparison(self, standard_metrics, adaptive_metrics, symbol=None, title=None):
        """
        Порівняння метрик стандартного та адаптивного підходів

        :param standard_metrics: Метрики стандартного підходу
        :param adaptive_metrics: Метрики адаптивного підходу
        :param symbol: Символ криптовалюти
        :param title: Заголовок графіка
        """
        # Збір метрик
        metrics = ['precision', 'recall', 'f1_score']

        std_values = [standard_metrics.get(m, 0) for m in metrics]
        adp_values = [adaptive_metrics.get(m, 0) for m in metrics]

        # Створення графіка
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Графік порівняння метрик
        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width / 2, std_values, width, label='Стандартний', color='blue', alpha=0.7)
        ax1.bar(x + width / 2, adp_values, width, label='Адаптивний', color='red', alpha=0.7)

        # Додавання значень
        for i, v in enumerate(std_values):
            ax1.text(i - width / 2, v + 0.02, f"{v:.3f}", ha='center')

        for i, v in enumerate(adp_values):
            ax1.text(i + width / 2, v + 0.02, f"{v:.3f}", ha='center')

        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel('Значення')
        ax1.set_title('Порівняння метрик')
        ax1.legend()
        ax1.grid(True, axis='y')

        # Графік покращення
        improvement = {}
        for metric in metrics:
            std_value = standard_metrics.get(metric, 0)
            adp_value = adaptive_metrics.get(metric, 0)
            if std_value > 0:
                improvement[metric] = ((adp_value - std_value) / std_value) * 100
            else:
                improvement[metric] = 0

        ax2.bar(metrics, [improvement.get(m, 0) for m in metrics], color='green', alpha=0.7)

        # Додавання значень
        for i, m in enumerate(metrics):
            imp_value = improvement.get(m, 0)
            color = 'green' if imp_value >= 0 else 'red'
            ax2.text(i, imp_value + 2, f"{imp_value:.2f}%", ha='center', color=color)

        ax2.set_ylabel('Покращення (%)')
        ax2.set_title('Відносне покращення метрик')
        ax2.grid(True, axis='y')

        plt.suptitle(title or f'Порівняння метрик для {symbol}')
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, standard_cm, adaptive_cm, title=None):
        """
        Візуалізація матриць плутанини

        :param standard_cm: Матриця плутанини стандартного підходу
        :param adaptive_cm: Матриця плутанини адаптивного підходу
        :param title: Заголовок графіка
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Стандартна матриця плутанини
        sns.heatmap(standard_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Не подія', 'Подія'], yticklabels=['Не подія', 'Подія'], ax=ax1)
        ax1.set_title('Стандартний підхід')
        ax1.set_ylabel('Справжні мітки')
        ax1.set_xlabel('Передбачені мітки')

        # Адаптивна матриця плутанини
        sns.heatmap(adaptive_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Не подія', 'Подія'], yticklabels=['Не подія', 'Подія'], ax=ax2)
        ax2.set_title('Адаптивний підхід')
        ax2.set_ylabel('Справжні мітки')
        ax2.set_xlabel('Передбачені мітки')

        plt.suptitle(title or 'Порівняння матриць плутанини')
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, y_true, standard_probs, adaptive_probs, title=None):
        """
        Візуалізація ROC-кривих

        :param y_true: Справжні мітки
        :param standard_probs: Ймовірності стандартного підходу
        :param adaptive_probs: Ймовірності адаптивного підходу
        :param title: Заголовок графіка
        """
        # Розрахунок ROC-кривих
        fpr_std, tpr_std, _ = roc_curve(y_true, standard_probs)
        roc_auc_std = auc(fpr_std, tpr_std)

        fpr_adp, tpr_adp, _ = roc_curve(y_true, adaptive_probs)
        roc_auc_adp = auc(fpr_adp, tpr_adp)

        # Створення графіка
        plt.figure(figsize=(10, 8))

        plt.plot(fpr_std, tpr_std, color='blue', lw=2,
                 label=f'Стандартний (AUC = {roc_auc_std:.3f})')
        plt.plot(fpr_adp, tpr_adp, color='red', lw=2,
                 label=f'Адаптивний (AUC = {roc_auc_adp:.3f})')

        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title or 'Порівняння ROC-кривих')
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_weights_comparison(self, default_weights, optimized_weights, category=None, title=None):
        """
        Порівняння початкових та оптимізованих ваг сигналів

        :param default_weights: Початкові ваги
        :param optimized_weights: Оптимізовані ваги
        :param category: Категорія токена
        :param title: Заголовок графіка
        """
        # Збір всіх унікальних сигналів
        all_signals = set(list(default_weights.keys()) + list(optimized_weights.keys()))
        signals = sorted(all_signals)

        # Отримання ваг для кожного сигналу
        def_values = [default_weights.get(s, 0) for s in signals]
        opt_values = [optimized_weights.get(s, 0) for s in signals]

        # Створення графіка
        plt.figure(figsize=(12, 8))

        # Позиції для груп стовпців
        x = np.arange(len(signals))
        width = 0.35

        plt.bar(x - width / 2, def_values, width, label='Початкові ваги', color='blue', alpha=0.7)
        plt.bar(x + width / 2, opt_values, width, label='Оптимізовані ваги', color='red', alpha=0.7)

        # Налаштування графіка
        plt.ylabel('Вага')
        plt.title(title or f'Порівняння ваг сигналів для категорії {category}')
        plt.xticks(x, signals, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, axis='y')

        # Додавання відсотків зміни
        for i, (def_val, opt_val) in enumerate(zip(def_values, opt_values)):
            if def_val > 0:
                percent_change = ((opt_val - def_val) / def_val) * 100
                color = 'green' if percent_change >= 0 else 'red'
                plt.text(i, max(def_val, opt_val) + 0.02,
                         f"{percent_change:.1f}%",
                         ha='center', color=color, fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_probability_distributions(self, standard_df, adaptive_df, title=None):
        """
        Візуалізація розподілів ймовірностей для стандартного та адаптивного підходів

        :param standard_df: DataFrame з результатами стандартного підходу
        :param adaptive_df: DataFrame з результатами адаптивного підходу
        :param title: Заголовок графіка
        """
        if standard_df.empty or adaptive_df.empty:
            print("Недостатньо даних для візуалізації розподілів")
            return

        # Створення графіка
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Перший графік: розподіли для подій
        std_events = standard_df[standard_df['is_event'] == 1]['standard_probability']
        adp_events = adaptive_df[adaptive_df['is_event'] == 1]['adaptive_probability']

        sns.histplot(std_events, bins=15, kde=True, color='blue', alpha=0.5,
                     label='Стандартний', ax=ax1)
        sns.histplot(adp_events, bins=15, kde=True, color='red', alpha=0.5,
                     label='Адаптивний', ax=ax1)

        ax1.set_title('Розподіл ймовірностей для подій')
        ax1.set_xlabel('Ймовірність')
        ax1.set_ylabel('Кількість')
        ax1.legend()
        ax1.grid(True)

        # Другий графік: розподіли для не-подій
        std_non_events = standard_df[standard_df['is_event'] == 0]['standard_probability']
        adp_non_events = adaptive_df[adaptive_df['is_event'] == 0]['adaptive_probability']

        sns.histplot(std_non_events, bins=15, kde=True, color='cyan', alpha=0.5,
                     label='Стандартний', ax=ax2)
        sns.histplot(adp_non_events, bins=15, kde=True, color='orange', alpha=0.5,
                     label='Адаптивний', ax=ax2)

        ax2.set_title('Розподіл ймовірностей для не-подій')
        ax2.set_xlabel('Ймовірність')
        ax2.set_ylabel('Кількість')
        ax2.legend()
        ax2.grid(True)

        plt.suptitle(title or 'Порівняння розподілів ймовірностей')
        plt.tight_layout()
        plt.show()

    def plot_accuracy_timeline(self, predictions_df, window_size=10, title=None):
        """
        Візуалізація зміни точності передбачення з часом

        :param predictions_df: DataFrame з результатами бектестингу
        :param window_size: Розмір ковзного вікна для розрахунку точності
        :param title: Заголовок графіка
        """
        if predictions_df.empty:
            print("Недостатньо даних для візуалізації зміни точності")
            return

        # Створення нових колонок для правильності передбачень
        if 'standard_prediction' in predictions_df.columns and 'adaptive_prediction' in predictions_df.columns:
            predictions_df['std_correct'] = (predictions_df['standard_prediction'] ==
                                             predictions_df['is_event']).astype(int)
            predictions_df['adp_correct'] = (predictions_df['adaptive_prediction'] ==
                                             predictions_df['is_event']).astype(int)
        else:
            predictions_df['std_correct'] = (predictions_df['prediction'] ==
                                             predictions_df['is_event']).astype(int)
            predictions_df['adp_correct'] = predictions_df['std_correct']

        # Розрахунок ковзного середнього для точності
        predictions_df['std_accuracy_ma'] = predictions_df['std_correct'].rolling(window=window_size).mean()
        predictions_df['adp_accuracy_ma'] = predictions_df['adp_correct'].rolling(window=window_size).mean()

        # Створення графіка
        plt.figure(figsize=(12, 6))

        plt.plot(predictions_df.index, predictions_df['std_accuracy_ma'], 'b-',
                 label='Стандартний (ковзна точність)')
        plt.plot(predictions_df.index, predictions_df['adp_accuracy_ma'], 'r-',
                 label='Адаптивний (ковзна точність)')

        # Додавання ліній трендів
        if len(predictions_df) > window_size:
            # Тренд для стандартного підходу
            valid_indices = ~np.isnan(predictions_df['std_accuracy_ma'])
            if sum(valid_indices) > 1:
                valid_data = predictions_df[valid_indices]
                z = np.polyfit(range(len(valid_data)), valid_data['std_accuracy_ma'], 1)
                p = np.poly1d(z)
                plt.plot(valid_data.index, p(range(len(valid_data))), 'b--', alpha=0.5,
                         label='Тренд (стандартний)')

            # Тренд для адаптивного підходу
            valid_indices = ~np.isnan(predictions_df['adp_accuracy_ma'])
            if sum(valid_indices) > 1:
                valid_data = predictions_df[valid_indices]
                z = np.polyfit(range(len(valid_data)), valid_data['adp_accuracy_ma'], 1)
                p = np.poly1d(z)
                plt.plot(valid_data.index, p(range(len(valid_data))), 'r--', alpha=0.5,
                         label='Тренд (адаптивний)')

        plt.title(title or f'Динаміка точності передбачень (ковзне вікно {window_size})')
        plt.ylabel('Точність (частка правильних)')
        plt.xlabel('Дата')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)

        # Форматування осі X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_threshold_optimization(self, y_true, y_prob, metric='f1', title=None):
        """
        Візуалізація оптимізації порогу для класифікації

        :param y_true: Справжні мітки
        :param y_prob: Ймовірності
        :param metric: Метрика для оптимізації ('f1', 'precision', 'recall')
        :param title: Заголовок графіка
        """
        from sklearn.metrics import precision_recall_curve, f1_score

        # Розрахунок precision, recall для різних порогів
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

        # Розрахунок F1-score для кожного порогу
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r > 0:  # Уникнення ділення на нуль
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0
            f1_scores.append(f1)

        # Створення графіка
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Графік precision-recall
        ax1.plot(recall, precision, color='blue', lw=2)
        ax1.set_title('Precision-Recall крива')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.grid(True)

        # Графік метрик в залежності від порогу
        ax2.plot(thresholds, precision[:-1], color='red', lw=2, label='Precision')
        ax2.plot(thresholds, recall[:-1], color='green', lw=2, label='Recall')
        ax2.plot(thresholds, f1_scores[:-1], color='blue', lw=2, label='F1-score')

        # Знаходження оптимального порогу за обраною метрикою
        if metric == 'f1':
            optimal_idx = np.argmax(f1_scores[:-1])
            optimal_value = f1_scores[optimal_idx]
            optimal_label = 'F1-score'
        elif metric == 'precision':
            optimal_idx = np.argmax(precision[:-1])
            optimal_value = precision[optimal_idx]
            optimal_label = 'Precision'
        elif metric == 'recall':
            optimal_idx = np.argmax(recall[:-1])
            optimal_value = recall[optimal_idx]
            optimal_label = 'Recall'
        else:
            raise ValueError(f"Непідтримувана метрика: {metric}")

        optimal_threshold = thresholds[optimal_idx]

        # Позначення оптимального порогу
        ax2.axvline(x=optimal_threshold, color='purple', linestyle='--', alpha=0.7)
        ax2.axhline(y=optimal_value, color='purple', linestyle='--', alpha=0.7)
        ax2.plot(optimal_threshold, optimal_value, 'ro', markersize=8)
        ax2.annotate(f'Оптимальний поріг: {optimal_threshold:.3f}\n{optimal_label}: {optimal_value:.3f}',
                     xy=(optimal_threshold, optimal_value),
                     xytext=(optimal_threshold + 0.1, optimal_value - 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

        ax2.set_title('Метрики в залежності від порогу')
        ax2.set_xlabel('Поріг')
        ax2.set_ylabel('Значення')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True)

        plt.suptitle(title or f'Оптимізація порогу за метрикою {metric}')
        plt.tight_layout()
        plt.show()

        return optimal_threshold, optimal_value
