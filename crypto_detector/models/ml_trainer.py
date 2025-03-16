import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

    def prepare_features(self, training_data):
        """
        Підготовка фічей та міток для навчання

        :param training_data: Дані для навчання
        :return: X (фічі), y (мітки)
        """
        if isinstance(training_data, str) and os.path.exists(training_data):
            # Якщо передано шлях до файлу
            data = pd.read_csv(training_data)
        elif isinstance(training_data, pd.DataFrame):
            # Якщо передано DataFrame
            data = training_data
        else:
            raise ValueError("training_data повинен бути шляхом до файлу або DataFrame")

        # Перевірка наявності необхідних колонок
        required_columns = ['is_event']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {required_columns}")

        # Розділення фічей та міток
        feature_columns = [col for col in data.columns if col not in ['is_event', 'timestamp', 'symbol']]
        X = data[feature_columns]
        y = data['is_event']

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
        # Розділення на тренувальні та тестові дані
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Визначення моделі та простору пошуку гіперпараметрів
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4, 5]
            }
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
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

        # Пошук оптимальних гіперпараметрів
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
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
        self.storage.save_model(best_model, model_name)

        return best_params, best_model

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

        # Проведення крос-валідації
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
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
        self.storage.save_model(pipeline, model_name)

        return report, pipeline

    def _get_feature_importance(self, pipeline, feature_names):
        """
        Отримання важливості фічей з моделі

        :param pipeline: Навчений конвеєр
        :param feature_names: Назви фічей
        :return: Словник з важливістю фічей
        """
        # Отримання моделі з конвеєра
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

    def visualize_feature_importance(self, feature_importance, title=None, top_n=20):
        """
        Візуалізація важливості фічей

        :param feature_importance: Словник з важливістю фічей
        :param title: Заголовок графіка (опціонально)
        :param top_n: Кількість найважливіших фічей для відображення
        """
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
