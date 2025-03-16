"""
Конфігураційний файл для налаштувань, констант та порогових значень
для системи виявлення підозрілої активності на криптовалютних ринках.
"""

# Базові налаштування
DEFAULT_EXCHANGE = 'binance'  # Біржа за замовчуванням
DEFAULT_TIMEFRAME = '5m'      # Таймфрейм за замовчуванням
DEFAULT_LOOKBACK_PERIOD = 24  # Період аналізу історичних даних в годинах
DATA_DIR = 'data'            # Директорія для зберігання даних

# Налаштування детектора
VOLUME_THRESHOLD_MULTIPLIER = 2.0   # Множник для виявлення незвичайного об'єму
PRICE_CHANGE_THRESHOLD = 3.0        # Поріг для виявлення значної зміни ціни (%)
VOLUME_CHANGE_THRESHOLD = 30.0      # Поріг для виявлення значної зміни об'єму (%)
ORDER_BOOK_RATIO_THRESHOLD = 1.3    # Поріг співвідношення ордерів купівлі/продажу
SOCIAL_CHANGE_THRESHOLD = 40.0      # Поріг для виявлення сплеску активності у соцмережах (%)
ALERT_THRESHOLD = 0.35             # Поріг для створення сповіщень

# Пороги за типами токенів
TOKEN_THRESHOLDS = {
    'meme': 0.12,         # Нижчий поріг для мем-токенів
    'defi': 0.15,         # Середній поріг для DeFi-токенів
    'l1_blockchain': 0.15, # Середній поріг для L1-блокчейнів
    'l2_scaling': 0.15,   # Середній поріг для L2-рішень
    'gaming': 0.15,       # Середній поріг для ігрових токенів
    'exchange': 0.18,     # Вищий поріг для токенів бірж
    'other': 0.18         # За замовчуванням для інших токенів
}

# Мапа категорій токенів
TOKEN_CATEGORIES = {
    'meme': ['PEPE', 'SHIB', 'DOGE', 'FLOKI', 'CAT', 'CTT', 'WIF'],
    'defi': ['UNI', 'AAVE', 'CAKE', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI'],
    'l1_blockchain': ['ETH', 'SOL', 'ADA', 'AVAX', 'DOT', 'NEAR', 'FTM', 'ATOM'],
    'l2_scaling': ['MATIC', 'ARB', 'OP', 'IMX', 'ZK', 'BASE', 'STX'],
    'gaming': ['AXS', 'SAND', 'MANA', 'ENJ', 'GALA', 'ILV'],
    'exchange': ['BNB', 'CRO', 'FTT', 'KCS', 'LEO', 'OKB']
}

# Початкові ваги сигналів
DEFAULT_SIGNAL_WEIGHTS = {
    'Аномальний обсяг торгів': 0.35,
    'Активна цінова динаміка': 0.25,
    'Дисбаланс книги ордерів': 0.20,
    'Підвищена активність у соціальних мережах': 0.20,
    'Нові лістинги на біржах': 0.10,
    'Підозрілий часовий патерн': 0.15,
    'Корельована активність з іншими монетами': 0.15,
    'Прискорення зростання об\'єму': 0.15
}

# Години підвищеного ризику (ніч та пізній вечір)
HIGH_RISK_HOURS = {0, 1, 2, 3, 4, 20, 21, 22, 23}

# Налаштування адаптивного навчання
LEARNING_RATE = 0.05           # Швидкість навчання
MIN_WEIGHT = 0.05              # Мінімальна вага сигналу
MAX_WEIGHT = 0.50              # Максимальна вага сигналу
TOTAL_MAX_WEIGHT = 1.5         # Максимальна сума ваг
TRAINING_THRESHOLD = 50        # Мінімальна кількість зразків для адаптації ваг
RETRAINING_INTERVAL = 100      # Інтервал для перенавчання ML моделей

# Налаштування бектестингу
BACKTEST_MIN_PRICE_CHANGE = 5.0  # Мінімальна зміна ціни для визначення pump-фази (%)
BACKTEST_MIN_DUMP_PERCENT = 5.0  # Мінімальний відсоток падіння після піку (%)
BACKTEST_WINDOW_HOURS = 24       # Вікно для аналізу в годинах

# Налаштування візуалізації
VISUALIZATION_COLORS = {
    'price': '#1f77b4',         # Синій для ціни
    'volume': '#2ca02c',        # Зелений для об'єму
    'event_line': '#d62728',    # Червоний для ліній подій
    'prediction': '#ff7f0e',    # Помаранчевий для передбачень
    'threshold': '#9467bd'      # Фіолетовий для порогової лінії
}

# Налаштування логування
LOG_LEVEL = 'INFO'              # Рівень логування (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE = 'detector.log'       # Файл для логування
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Формат логів

# Налаштування моніторингу
MONITORING_INTERVAL = 300       # Інтервал між перевірками в секундах (5 хвилин)
ALERTS_DIR = 'alerts'           # Директорія для збереження сповіщень

#Соцмережі
#Twitter
TWITTER_API_KEY = '8O8LdvJU2oGm03LUrd3BE3pqq'
TWITTER_API_SECRET = '4osSUrmD25uFl8YPC9Sy8P8mFTtDl8LD9W1tl3Df2dIMyududT'
TWITTER_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAI3ozwEAAAAAlGCrGGXRWBYOUMcVjqLMxdomZJk%3DYK4iXY2Rqoa1axE87ZBvbQIlhvyIhgbr9xnTK2NurWQh3pZr1a'

#Reddit
REDDIT_CLIENT_ID = 'gWSkeeqglKEzuw5tjTrIxQ'
REDDIT_CLIENT_SECRET = 'bBmFxSW2_yXfRcc72EBGM9mnM0EnKg'
REDDIT_USER_AGENT = 'crypto_detector/1.0'

#Telegram
TELEGRAM_API_ID = '8595206'
TELEGRAM_API_HASH = '1e07f65fafa218c92a43ea1a33ba990b'
