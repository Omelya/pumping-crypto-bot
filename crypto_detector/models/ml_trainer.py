import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import ast

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from crypto_detector.data.data_storage import DataStorage


class MLTrainer:
    """
    Клас для навчання та оцінки ML моделей для виявлення pump-and-dump схем.
    Забезпечує тренування, налаштування гіперпараметрів, оцінку та збереження моделей.
    """

    def __init__(self, storage=None):
        """
        Ініціалізація тренера ML моделей

        :param storage: Система зберігання даних
        """
        self.storage = storage or DataStorage()

        # Налаштування для навчання
        self.test_size = 0.3
        self.random_state = 42

        # Налаштування для логування
        self.training_history = {}

    def extract_signal_features(self, signals_str):
        """
        Витягує числові ознаки з рядкового представлення сигналів

        :param signals_str: Рядкове представлення списку сигналів
        :return: Словник з ознаками
        """
        if not isinstance(signals_str, str) or not signals_str:
            return {}

        try:
            # Очистка рядка від зайвих символів
            signals_str = signals_str.replace('\\"', '"')

            # Заміна одинарних лапок на подвійні для правильного парсингу JSON
            signals_str = signals_str.replace("'", '"')

            # Спроба розібрати сигнали як список словників
            try:
                signals = json.loads(signals_str)
            except json.JSONDecodeError:
                # Якщо не вдалося через JSON, спробуємо через ast
                try:
                    signals = ast.literal_eval(signals_str)
                except (SyntaxError, ValueError):
                    # Якщо і так не вдалося, повертаємо порожній словник
                    return {}

            # Витягуємо числові значення з опису сигналів
            features = {}

            for signal in signals:
                name = signal.get('name', '')
                weight = signal.get('weight', 0)
                description = signal.get('description', '')

                # Зберігаємо вагу сигналу
                features[f"{name.replace(' ', '_')}_weight"] = weight

                # Витягуємо числа з опису
                numbers = re.findall(r'-?\d+\.\d+|-?\d+', description)
                if numbers:
                    features[f"{name.replace(' ', '_')}_value"] = float(numbers[0])

            return features

        except Exception as e:
            print(f"Помилка при обробці сигналів: {str(e)}")
            return {}

    def prepare_features(self, training_data):
        """
        Підготовка фічей та міток для навчання

        :param training_data: Дані для навчання
        :return: X (фічі), y (мітки)
        """
        if isinstance(training_data, str) and os.path.exists(training_data):
            # Якщо передано шлях до файлу
            print(f"Завантаження даних з файлу: {training_data}")
            data = pd.read_csv(training_data)
        elif isinstance(training_data, pd.DataFrame):
            # Якщо передано DataFrame
            data = training_data.copy()
        else:
            raise ValueError("training_data повинен бути шляхом до файлу або DataFrame")

        # Перевірка наявності необхідних колонок
        required_columns = ['is_event']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {required_columns}")

        print(f"Початкові дані: {data.shape}, колонки: {data.columns.tolist()}")

        # Перетворюємо часову мітку в числові ознаки (година доби, день тижня)
        if 'timestamp' in data.columns:
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data['hour_of_day'] = data['timestamp'].dt.hour
                data['day_of_week'] = data['timestamp'].dt.dayofweek
                data = data.drop('timestamp', axis=1)
                print("Успішно перетворено timestamp у числові ознаки")
            except Exception as e:
                print(f"Помилка при обробці timestamp: {str(e)}")
                if 'timestamp' in data.columns:
                    data = data.drop('timestamp', axis=1)

        # Обробка сигналів, якщо вони є
        if 'signals' in data.columns:
            try:
                # Витягуємо ознаки з сигналів
                print("Обробка колонки 'signals'...")
                signal_features_list = []

                for idx, row in data.iterrows():
                    signal_features = self.extract_signal_features(row['signals'])
                    signal_features_list.append(signal_features)

                # Створюємо DataFrame з цих ознак
                signal_df = pd.DataFrame(signal_features_list)

                # Об'єднуємо з основними даними, якщо є що об'єднувати
                if not signal_df.empty:
                    print(f"Витягнуто {signal_df.shape[1]} ознак з сигналів")
                    data = pd.concat([data, signal_df], axis=1)
                else:
                    print("Не вдалося витягнути ознаки з сигналів")

                # Видаляємо оригінальну колонку
                data = data.drop('signals', axis=1)
            except Exception as e:
                print(f"Помилка при обробці сигналів: {str(e)}")
                # Просто видаляємо колонку, якщо не вдалося обробити
                data = data.drop('signals', axis=1)

        # Те саме для adaptive_signals, якщо є
        if 'adaptive_signals' in data.columns:
            try:
                print("Обробка колонки 'adaptive_signals'...")
                adaptive_signal_features_list = []

                for idx, row in data.iterrows():
                    signal_features = self.extract_signal_features(row['adaptive_signals'])
                    # Додаємо префікс, щоб відрізняти від звичайних сигналів
                    signal_features = {f"adaptive_{k}": v for k, v in signal_features.items()}
                    adaptive_signal_features_list.append(signal_features)

                # Створюємо DataFrame з цих ознак
                adaptive_signal_df = pd.DataFrame(adaptive_signal_features_list)

                # Об'єднуємо з основними даними, якщо є що об'єднувати
                if not adaptive_signal_df.empty:
                    print(f"Витягнуто {adaptive_signal_df.shape[1]} ознак з adaptive_signals")
                    data = pd.concat([data, adaptive_signal_df], axis=1)
                else:
                    print("Не вдалося витягнути ознаки з adaptive_signals")

                # Видаляємо оригінальну колонку
                data = data.drop('adaptive_signals', axis=1)
            except Exception as e:
                print(f"Помилка при обробці adaptive_signals: {str(e)}")
                # Просто видаляємо колонку, якщо не вдалося обробити
                data = data.drop('adaptive_signals', axis=1)

        # Видаляємо символ, якщо він є
        if 'symbol' in data.columns:
            data = data.drop('symbol', axis=1)

        # Переконуємось, що всі колонки, окрім is_event, є числового типу
        feature_columns = [col for col in data.columns if col != 'is_event']

        for col in feature_columns:
            # Переконуємося, що колонка має числовий тип
            if data[col].dtype == 'object':
                print(f"Конвертація нечислової колонки '{col}' у числовий формат")
                # Спробуйте конвертувати у числовий тип
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Заповнюємо пропущені значення нулями
        print("Заповнення пропущених значень нулями")
        data = data.fillna(0)

        # Переконуємося, що всі дані числові
        non_numeric_columns = []
        for col in data.columns:
            if col != 'is_event' and data[col].dtype == 'object':
                non_numeric_columns.append(col)

        if non_numeric_columns:
            print(f"Увага! Колонки {non_numeric_columns} все ще мають нечисловий тип після обробки!")
            # Видаляємо проблемні колонки
            data = data.drop(non_numeric_columns, axis=1)
            # Оновлюємо список колонок для ознак
            feature_columns = [col for col in data.columns if col != 'is_event']

        # Створюємо X та y
        X = data[feature_columns]
        y = data['is_event']

        print(f"Підготовлено {len(X)} зразків з {len(feature_columns)} ознаками")

        return X, y

    def train_model(self, X, y, model_type='gradient_boosting', category=None, **kwargs):
        """
        Навчання ML моделі

        :param X: Фічі
        :param y: Мітки
        :param model_type: Тип моделі ('gradient_boosting' або 'random_forest')
        :param category: Категорія токена (опціонально)
        :param kwargs: Додаткові параметри для моделі
        :return: Навчена модель
        """
        # Розділення на тренувальні та тестові дані
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Створення конвеєра з нормалізацією та моделлю
        pipeline_steps = [
            ('scaler', StandardScaler())
        ]

        # Вибір моделі в залежності від типу
        if model_type == 'gradient_boosting':
            # Параметри за замовчуванням або з kwargs
            n_estimators = kwargs.get('n_estimators', 100)
            learning_rate = kwargs.get('learning_rate', 0.1)
            max_depth = kwargs.get('max_depth', 3)

            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=self.random_state
            )
        elif model_type == 'random_forest':
            # Параметри за замовчуванням або з kwargs
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}")

        # Додавання моделі до конвеєра
        pipeline_steps.append(('model', model))
        pipeline = Pipeline(pipeline_steps)

        # Навчання моделі
        pipeline.fit(X_train, y_train)

        # Оцінка моделі
        y_pred = pipeline.predict(X_test)

        # Розрахунок метрик
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Розрахунок ROC AUC, якщо є predict_proba
        roc_auc = 0
        if hasattr(pipeline, 'predict_proba'):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)

        # Збереження результатів
        model_info = {
            'model_type': model_type,
            'category': category,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self._get_feature_importance(pipeline, X.columns),
            'timestamp': datetime.now().isoformat(),
            'params': kwargs
        }

        # Збереження в історії
        model_id = f"{model_type}_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.training_history[model_id] = model_info

        print(
            f"Модель навчена успішно. Метрики: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")

        return pipeline, model_info

    def optimize_hyperparameters(self, X, y, model_type='gradient_boosting', category=None):
        """
        Оптимізація гіперпараметрів моделі

        :param X: Фічі
        :param y: Мітки
        :param model_type: Тип моделі ('gradient_boosting' або 'random_forest')
        :param category: Категорія токена (опціонально)
        :return: Найкращі параметри та модель
        """
        # Перевіримо, чи маємо достатньо даних
        if len(X) < 50:
            print(
                f"Недостатньо даних для оптимізації гіперпараметрів (маємо {len(X)} зразків). Використовуємо стандартні параметри.")
            if model_type == 'gradient_boosting':
                return {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}, None
            else:
                return {'n_estimators': 100, 'max_depth': None}, None

        # Розділення на тренувальні та тестові дані
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Визначення моделі та простору пошуку гіперпараметрів
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=self.random_state)
            # Використовуємо менше параметрів для прискорення
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [2, 3]
            }
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 5],
                'min_samples_split': [2, 5]
            }
        else:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}")

        # Створення конвеєра з нормалізацією та моделлю
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Відповідно модифікуємо параметри для GridSearchCV
        param_grid = {f'model__{k}': v for k, v in param_grid.items()}

        # Пошук оптимальних гіперпараметрів (зменшена кількість crossval для прискорення)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )

        try:
            grid_search.fit(X_train, y_train)

            # Отримання найкращих параметрів
            best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
            best_score = grid_search.best_score_

            print(f"Найкращі параметри: {best_params}")
            print(f"Найкращий F1-score: {best_score:.4f}")

            # Оцінка на тестовому наборі
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Розрахунок метрик
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"Оцінка на тестовому наборі: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

            # Збереження моделі з оптимальними параметрами
            model_name = f"{model_type}_{category}_optimized" if category else f"{model_type}_optimized"
            if self.storage:
                self.storage.save_model(best_model, model_name)

            return best_params, best_model

        except Exception as e:
            print(f"Помилка при оптимізації гіперпараметрів: {str(e)}")
            print("Використовуємо стандартні параметри:")
            if model_type == 'gradient_boosting':
                best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            else:
                best_params = {'n_estimators': 100, 'max_depth': None}

            print(f"Стандартні параметри: {best_params}")
            return best_params, None

    def cross_validate_and_report(self, X, y, model_type='gradient_boosting', category=None, **kwargs):
        """
        Проведення крос-валідації та звітування про продуктивність моделі

        :param X: Фічі
        :param y: Мітки
        :param model_type: Тип моделі ('gradient_boosting' або 'random_forest')
        :param category: Категорія токена (опціонально)
        :param kwargs: Додаткові параметри для моделі
        :return: Звіт з результатами
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        # Перевіримо, чи маємо достатньо даних
        if len(X) < 30:
            print(f"Недостатньо даних для крос-валідації (маємо {len(X)} зразків). Пропускаємо.")
            empty_report = {
                'model_type': model_type,
                'category': category,
                'cross_validation': {
                    'accuracy': {'mean': 0, 'std': 0, 'values': []},
                    'precision': {'mean': 0, 'std': 0, 'values': []},
                    'recall': {'mean': 0, 'std': 0, 'values': []},
                    'f1': {'mean': 0, 'std': 0, 'values': []}
                },
                'feature_importance': {},
                'timestamp': datetime.now().isoformat(),
                'params': kwargs
            }
            return empty_report, None

        # Створення конвеєра
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=self.random_state
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Непідтримуваний тип моделі: {model_type}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Проведення крос-валідації (зменшена кількість folds для прискорення)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        try:
            cv_scores = {
                'accuracy': cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy'),
                'precision': cross_val_score(pipeline, X, y, cv=cv, scoring='precision'),
                'recall': cross_val_score(pipeline, X, y, cv=cv, scoring='recall'),
                'f1': cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
            }

            # Навчання фінальної моделі на всіх даних
            pipeline.fit(X, y)

            # Генерація звіту
            report = {
                'model_type': model_type,
                'category': category,
                'cross_validation': {
                    'accuracy': {
                        'mean': cv_scores['accuracy'].mean(),
                        'std': cv_scores['accuracy'].std(),
                        'values': cv_scores['accuracy'].tolist()
                    },
                    'precision': {
                        'mean': cv_scores['precision'].mean(),
                        'std': cv_scores['precision'].std(),
                        'values': cv_scores['precision'].tolist()
                    },
                    'recall': {
                        'mean': cv_scores['recall'].mean(),
                        'std': cv_scores['recall'].std(),
                        'values': cv_scores['recall'].tolist()
                    },
                    'f1': {
                        'mean': cv_scores['f1'].mean(),
                        'std': cv_scores['f1'].std(),
                        'values': cv_scores['f1'].tolist()
                    }
                },
                'feature_importance': self._get_feature_importance(pipeline, X.columns),
                'timestamp': datetime.now().isoformat(),
                'params': kwargs
            }

            # Виведення звіту
            print(f"Результати крос-валідації для {model_type}:")
            for metric, values in cv_scores.items():
                print(f"{metric.capitalize()}: {values.mean():.4f} ± {values.std():.4f}")

            # Збереження моделі
            model_name = f"{model_type}_{category}_cv" if category else f"{model_type}_cv"
            if self.storage:
                self.storage.save_model(pipeline, model_name)

            return report, pipeline

        except Exception as e:
            print(f"Помилка при крос-валідації: {str(e)}")
            # Спростимо процес при виникненні помилок
            try:
                # Просто навчимо модель на всіх даних
                pipeline.fit(X, y)

                report = {
                    'model_type': model_type,
                    'category': category,
                    'cross_validation': {
                        'accuracy': {'mean': 0, 'std': 0, 'values': []},
                        'precision': {'mean': 0, 'std': 0, 'values': []},
                        'recall': {'mean': 0, 'std': 0, 'values': []},
                        'f1': {'mean': 0, 'std': 0, 'values': []}
                    },
                    'feature_importance': self._get_feature_importance(pipeline, X.columns),
                    'timestamp': datetime.now().isoformat(),
                    'params': kwargs
                }

                # Збереження моделі
                model_name = f"{model_type}_{category}_cv" if category else f"{model_type}_cv"
                if self.storage:
                    self.storage.save_model(pipeline, model_name)

                return report, pipeline
            except Exception as inner_e:
                print(f"Помилка при навчанні моделі: {str(inner_e)}")
                return {
                    'model_type': model_type,
                    'category': category,
                    'error': str(inner_e),
                    'timestamp': datetime.now().isoformat(),
                    'params': kwargs
                }, None

    def _get_feature_importance(self, pipeline, feature_names):
        """
        Отримання важливості фічей з моделі

        :param pipeline: Навчений конвеєр
        :param feature_names: Назви фічей
        :return: Словник з важливістю фічей
        """
        # Отримання моделі з конвеєра
        try:
            model = pipeline.named_steps['model']

            # Перевірка наявності атрибуту feature_importances_
            if hasattr(model, 'feature_importances_'):
                # Створення словника {назва_фічі: важливість}
                importance_dict = dict(zip(feature_names, model.feature_importances_))

                # Сортування за спаданням важливості
                sorted_importance = {k: v for k, v in
                                     sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}

                return sorted_importance
            else:
                return {}
        except Exception as e:
            print(f"Помилка при отриманні важливості фічей: {str(e)}")
            return {}

    def visualize_feature_importance(self, feature_importance, title=None, top_n=20):
        """
        Візуалізація важливості фічей

        :param feature_importance: Словник з важливістю фічей
        :param title: Заголовок графіка (опціонально)
        :param top_n: Кількість найважливіших фічей для відображення
        """
        if not feature_importance:
            print("Немає даних для візуалізації важливості фічей")
            return

        # Сортування за важливістю
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])

        # Створення DataFrame для візуалізації
        df = pd.DataFrame({'Feature': list(sorted_features.keys()), 'Importance': list(sorted_features.values())})

        # Створення графіка
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df)

        # Налаштування графіка
        plt.title(title or 'Feature Importance')
        plt.tight_layout()
        plt.show()

    def visualize_confusion_matrix(self, confusion_matrix_data, title=None):
        """
        Візуалізація матриці плутанини

        :param confusion_matrix_data: Матриця плутанини
        :param title: Заголовок графіка (опціонально)
        """
        # Створення графіка
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Не подія', 'Подія'], yticklabels=['Не подія', 'Подія'])

        # Налаштування графіка
        plt.title(title or 'Confusion Matrix')
        plt.ylabel('Справжні мітки')
        plt.xlabel('Передбачені мітки')
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

    def save_training_history(self, filename='training_history.json'):
        """
        Збереження історії навчання

        :param filename: Ім'я файлу для збереження
        :return: Шлях до збереженого файлу
        """
        # Шлях до файлу
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models', filename)

        # Збереження історії у файл
        import json
        with open(file_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)

        print(f"Історію навчання збережено у {file_path}")
        return file_path
