from telethon import TelegramClient
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# Налаштуйте ці значення
API_ID = os.getenv('TELEGRAM_API_ID', '')
API_HASH = os.getenv('TELEGRAM_API_HASH', '')
PHONE = os.getenv('TELEGRAM_PHONE', '')
SESSION_FILE = 'telegram_session'  # Назва файлу сесії, яку потрібно зберегти


async def create_session():
    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    await client.connect()

    if not await client.is_user_authorized():
        # Запитуємо код аутентифікації
        await client.send_code_request(PHONE)
        code = input(f"Введіть код, надісланий в Telegram на {PHONE}: ")

        try:
            # Виконуємо вхід
            await client.sign_in(PHONE, code)
        except Exception as e:
            # Якщо увімкнена двофакторна автентифікація
            if "2FA" in str(e) or "password" in str(e):
                password = input("Введіть ваш пароль двофакторної автентифікації: ")
                await client.sign_in(password=password)
            else:
                raise e

    print(f"Авторизація успішна! Файл сесії збережено як '{SESSION_FILE}'")

    # Отримаємо та відобразимо інформацію про обліковий запис для підтвердження
    me = await client.get_me()
    print(f"Ви увійшли як: {me.first_name} {me.last_name or ''} (@{me.username or 'без імені користувача'})")

    await client.disconnect()


asyncio.run(create_session())
