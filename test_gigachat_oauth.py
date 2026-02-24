"""
Простой тест GigaChat: получить токен по ключу и запросить список моделей.
Запуск: python test_gigachat_oauth.py
"""
import httpx
import uuid

# Ваш authorization_key (Base64 от client_id:client_secret)
AUTHORIZATION_KEY = "M2RjNGFkZGEtOTA0MS00MzI0LTlmNzUtNzczNTIxNmQ0Zjk1OmFmNzE0NWQ3LWY5NDQtNGExNC05ZmZmLWEzYjE3Zjk5MjgwYw=="

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
MODELS_URL = "https://gigachat.devices.sberbank.ru/api/v1/models"
CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
TIMEOUT = 30
VERIFY_SSL = False


def main():
    print("1. Запрос токена OAuth...")
    try:
        with httpx.Client(timeout=TIMEOUT, verify=VERIFY_SSL) as client:
            r = client.post(
                OAUTH_URL,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4()),
                    "Authorization": f"Basic {AUTHORIZATION_KEY}",
                },
                data={"scope": "GIGACHAT_API_PERS"},
            )
        print(f"   Статус: {r.status_code}")
        print(f"   Ответ: {r.text[:500]}{'...' if len(r.text) > 500 else ''}")
        r.raise_for_status()
        data = r.json()
        access_token = data.get("access_token")
        if not access_token:
            print("   Ошибка: в ответе нет access_token. Полный ответ:", data)
            return
        print("   Токен получен.")
    except Exception as e:
        print(f"   Ошибка: {e}")
        return

    print("\n2. Запрос списка моделей (GET /api/v1/models)...")
    try:
        with httpx.Client(timeout=TIMEOUT, verify=VERIFY_SSL) as client:
            r = client.get(
                MODELS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        print(f"   Статус: {r.status_code}")
        print(f"   Ответ: {r.text[:800]}{'...' if len(r.text) > 800 else ''}")
        r.raise_for_status()
        print("   Успех.")
    except Exception as e:
        print(f"   Ошибка: {e}")

    print("\n3. Запрос в чат: «Как дела?» (POST /api/v1/chat/completions)...")
    try:
        with httpx.Client(timeout=TIMEOUT, verify=VERIFY_SSL) as client:
            r = client.post(
                CHAT_URL,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {access_token}",
                },
                json={
                    "model": "GigaChat-2-Max",
                    "messages": [{"role": "user", "content": "Как дела?"}],
                    "stream": False,
                    "repetition_penalty": 1,
                },
            )
        print(f"   Статус: {r.status_code}")
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            print(f"   Ответ модели: {content}")
        else:
            print(f"   Ответ: {data}")
    except Exception as e:
        print(f"   Ошибка: {e}")


if __name__ == "__main__":
    main()
