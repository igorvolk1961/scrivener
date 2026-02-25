"""
Простейший тест: запрос «Как дела» к GigaChat с интерфейсом OpenAI
(client.chat.completions.create → response.choices[0].message.content).

При прямом обращении (библиотека gigachat или HTTP к GigaChat) используется
сырой токен без префиксов (Bearer <токен>). Префиксы giga-auth-, giga-cred-,
giga-user- нужны только при вызове через gpt2giga (прокси по ним понимает,
что передавать в GigaChat).

Сервис gpt2giga при необходимости запускайте вручную, например:
  gpt2giga --proxy.pass-token true
При ошибке SSL (self-signed certificate) добавьте:
  --gigachat.verify-ssl-certs false
"""

import asyncio
import os
import sys
import uuid
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from openai import AsyncOpenAI

from api.services.gpt2giga_runner import get_gpt2giga_base_url

# --- Подставьте вручную для отладки ---
API_KEY_OR_TOKEN = "M2RjNGFkZGEtOTA0MS00MzI0LTlmNzUtNzczNTIxNmQ0Zjk1OmFmNzE0NWQ3LWY5NDQtNGExNC05ZmZmLWEzYjE3Zjk5MjgwYw=="

USE_TOKEN = False   # True — считать API_KEY_OR_TOKEN токеном доступа
FETCH_TOKEN = True  # при ключе (base64) получить временный токен через OAuth

# True — в этом процессе вызывать библиотеку gigachat напрямую (GigaChat.achat);
# False — только HTTP в прокси gpt2giga (библиотеку не импортируем; прокси сам дергает GigaChat).
USE_GIGACHAT_LIB_DIRECT = True
# True — один прогон: получить токен один раз, вызвать прямой achat и запрос через прокси с тем же токеном (для сравнения).
COMPARE_BOTH = True
# Параметры, как в прокси (gigachat_settings.model_dump()). Исключите из списка те, что не передавать — посмотрим, какой ломает.
EXCLUDE_PROXY_PARAMS: tuple[str, ...] = ()  # например ("auth_url", "scope") или ("auth_url",) чтобы проверить
# True — в COMPARE_BOTH добавить шаг: через прокси с giga-cred-<ключ> (OAuth по ключу в прокси, без получения токена в тесте).
USE_CREDENTIALS_FOR_PROXY = True
# ---

GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
OAUTH_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

# Те же параметры, что подставляет прокси из gigachat_settings (из лога прокси). pass_token потом ставит access_token и сбрасывает credentials/user/password.
PROXY_LIKE_GIGACHAT_SETTINGS = {
    "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
    "auth_url": "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
    "scope": "GIGACHAT_API_PERS",
    "model": None,
    "profanity_check": None,
    "user": None,
    "timeout": 30.0,
    "verify_ssl_certs": False,
    "ssl_context": None,
    "ca_bundle_file": None,
    "cert_file": None,
    "key_file": None,
    "flags": None,
    "max_connections": None,
    "max_retries": 0,
    "retry_backoff_factor": 0.5,
    "retry_on_status_codes": (429, 500, 502, 503, 504),
    "token_expiry_buffer_ms": 60000,
}


def _auth_fingerprint(value: str | None, name: str = "token") -> str:
    """Для сравнения значений токена/ключа без вывода в лог: len, первые и последние 4 символа."""
    if not value:
        return f"{name}: None/empty"
    if len(value) <= 8:
        return f"{name}: len={len(value)} (short)"
    return f"{name}: len={len(value)} first={value[:4]!r} last={value[-4:]!r}"


def get_access_token(credentials: str, scope: str = GIGACHAT_SCOPE) -> str:
    """Получение временного токена доступа по ключу авторизации (OAuth2)."""
    credentials = credentials.strip()
    if not credentials:
        raise ValueError("Ключ авторизации (base64) не задан")
    headers = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    payload = {"scope": scope}
    with httpx.Client(timeout=60, verify=False) as client:
        response = client.post(OAUTH_TOKEN_URL, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
    # OAuth может вернуть access_token или tok (формат GigaChat)
    if "access_token" in data:
        return data["access_token"]
    if "tok" in data:
        return data["tok"]
    raise RuntimeError(f"Токен не найден в ответе OAuth. Поля: {list(data.keys())}")


class GigaChatOpenAIWrapper:
    """Обёртка с интерфейсом OpenAI: вызов GigaChat API (Bearer + RqUID), ответ в формате OpenAI."""

    def __init__(self, token: str):
        self._token = token
        self._url = GIGACHAT_CHAT_URL

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, *, model: str, messages: list, max_tokens: int = 500, **kwargs):
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            r = await client.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )
        r.raise_for_status()
        data = r.json()
        # Формируем объект в стиле OpenAI response
        choices = []
        if data.get("choices"):
            c = data["choices"][0]
            msg = c.get("message", {})
            choices.append(SimpleNamespace(message=SimpleNamespace(content=msg.get("content") or "")))
        return SimpleNamespace(choices=choices)


async def main() -> None:
    raw = API_KEY_OR_TOKEN.strip()
    if not raw:
        print("Задайте API_KEY_OR_TOKEN в начале файла (ключ base64 или токен).")
        return

    # Токен для прямых вызовов (библиотека / API) — всегда без префиксов
    token = None
    # Для gpt2giga нужен формат в заголовке: giga-auth-<токен> или giga-cred-<ключ>:<scope>
    api_key_gpt2giga = None

    if raw.startswith(("giga-auth-", "giga-cred-", "giga-user-")):
        api_key_gpt2giga = raw
        if raw.startswith("giga-auth-"):
            token = raw.replace("giga-auth-", "", 1)
    elif FETCH_TOKEN and not USE_TOKEN:
        print("Получение временного токена по ключу авторизации...")
        try:
            token = get_access_token(raw, GIGACHAT_SCOPE)
            api_key_gpt2giga = f"giga-auth-{token}"  # для gpt2giga — префикс только здесь
            print("Токен получен.")
        except Exception as e:
            print(f"Ошибка получения токена: {e}")
            return
    elif USE_TOKEN:
        token = raw
        api_key_gpt2giga = f"giga-auth-{raw}"
    else:
        api_key_gpt2giga = f"giga-cred-{raw}:{GIGACHAT_SCOPE}"

    # Режим сравнения: один токен — сначала прямой вызов, затем через прокси
    if token and COMPARE_BOTH:
        print("=== Один токен, оба пути (сравнение) ===")
        print("  [сравнение] Токен (один и тот же): ", _auth_fingerprint(token))
        # 1) Прямой вызов библиотеки
        print("\n1) Прямой вызов (библиотека GigaChat)...")
        try:
            from gigachat import GigaChat
            giga = GigaChat(access_token=token, verify_ssl_certs=False)
            response = await giga.achat("Как дела")
            content = response.choices[0].message.content if response.choices else ""
            print("   OK:", (content or "(пусто)")[:80])
        except Exception as e:
            print(f"   Ошибка: {e}")
        # 2) Прямой вызов с параметрами как в прокси (какой параметр ломает — исключите в EXCLUDE_PROXY_PARAMS)
        print("\n2) Прямой вызов с параметрами как в прокси (исключены: {})...".format(EXCLUDE_PROXY_PARAMS or "нет"))
        kwargs = {k: v for k, v in PROXY_LIKE_GIGACHAT_SETTINGS.items() if k not in EXCLUDE_PROXY_PARAMS}
        kwargs["access_token"] = token
        kwargs["credentials"] = None
        kwargs["password"] = None
        try:
            from gigachat import GigaChat
            giga = GigaChat(**kwargs)
            response = await giga.achat("Как дела")
            content = response.choices[0].message.content if response.choices else ""
            print("   OK:", (content or "(пусто)")[:80])
        except Exception as e:
            print(f"   Ошибка: {e}")

        # 2b) То же, что 2, но с RqUID в custom_headers_cvar (как в прокси) — проверка, ломает ли контекст
        print("\n2b) Прямой вызов (параметры как в прокси) + custom_headers_cvar RqUID как в прокси...")
        try:
            from gigachat import GigaChat
            from gigachat.context import custom_headers_cvar
            rquid = str(uuid.uuid4())
            token_ctx = custom_headers_cvar.set({"RqUID": rquid})
            try:
                giga = GigaChat(**{k: v for k, v in PROXY_LIKE_GIGACHAT_SETTINGS.items() if k not in EXCLUDE_PROXY_PARAMS}, access_token=token, credentials=None, password=None)
                response = await giga.achat("Как дела")
                content = response.choices[0].message.content if response.choices else ""
                print("   OK:", (content or "(пусто)")[:80])
            finally:
                custom_headers_cvar.reset(token_ctx)
        except Exception as e:
            print(f"   Ошибка: {e}")

        # 2c) Как в прокси: создать клиент БЕЗ access_token в конструкторе, потом подставить токен через _settings
        print("\n2c) Как в прокси: GigaChat(**params без токена), потом _settings.access_token = token, credentials = None...")
        try:
            from gigachat import GigaChat
            kwargs = {k: v for k, v in PROXY_LIKE_GIGACHAT_SETTINGS.items() if k not in EXCLUDE_PROXY_PARAMS}
            # В конструктор не передаём access_token (как config.model_dump() в прокси)
            giga = GigaChat(**kwargs)
            giga._settings.credentials = None
            giga._settings.user = None
            giga._settings.password = None
            giga._settings.access_token = token
            response = await giga.achat("Как дела")
            content = response.choices[0].message.content if response.choices else ""
            print("   OK:", (content or "(пусто)")[:80])
        except Exception as e:
            print(f"   Ошибка: {e}")

        # 3) Через прокси (тот же token в giga-auth-<token>)
        print("\n3) Через прокси gpt2giga (тот же токен, giga-auth-)...")
        try:
            base_url = get_gpt2giga_base_url().rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            client = AsyncOpenAI(base_url=base_url, api_key=f"giga-auth-{token}")
            response = await client.chat.completions.create(
                model="GigaChat",
                messages=[{"role": "user", "content": "Как дела"}],
                max_tokens=500,
            )
            content = response.choices[0].message.content if response.choices else ""
            print("   OK:", (content or "(пусто)")[:80])
        except Exception as e:
            print(f"   Ошибка: {e}")

        # 4) Через прокси с OAuth по ключу (giga-cred-<ключ>:scope) — токен не получаем, прокси сам сделает OAuth
        if USE_CREDENTIALS_FOR_PROXY and not raw.startswith(("giga-auth-", "giga-cred-", "giga-user-")):
            print("\n4) Через прокси gpt2giga с OAuth по ключу (giga-cred-)...")
            try:
                base_url = get_gpt2giga_base_url().rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url}/v1"
                api_key_cred = f"giga-cred-{raw}:{GIGACHAT_SCOPE}"
                client = AsyncOpenAI(base_url=base_url, api_key=api_key_cred)
                response = await client.chat.completions.create(
                    model="GigaChat",
                    messages=[{"role": "user", "content": "Как дела"}],
                    max_tokens=500,
                )
                content = response.choices[0].message.content if response.choices else ""
                print("   OK:", (content or "(пусто)")[:80])
            except Exception as e:
                print(f"   Ошибка: {e}")
        return

    # Единственная ветка, где в этом процессе используется библиотека gigachat (без сравнения)
    if token and USE_GIGACHAT_LIB_DIRECT:
        print("Отправка запроса «Как дела» через библиотеку GigaChat (в этом процессе, токен без префикса)...")
        print("  [сравнение] Прямой вызов, access_token: ", _auth_fingerprint(token))
        try:
            from gigachat import GigaChat  # только здесь импорт библиотеки
            giga = GigaChat(access_token=token, verify_ssl_certs=False)
            response = await giga.achat("Как дела")
            content = response.choices[0].message.content if response.choices else ""
            print("Ответ GigaChat:")
            print(content or "(пусто)")
        except Exception as e:
            print(f"Ошибка: {e}")
        return

    # Без библиотеки в этом процессе: либо прокси (gpt2giga), либо прямой HTTP к API GigaChat
    if USE_GIGACHAT_LIB_DIRECT and token:
        client = GigaChatOpenAIWrapper(token)
        print("Отправка запроса «Как дела» в GigaChat (Bearer + RqUID, без прокси)...")
    else:
        # Через gpt2giga — префиксы нужны только здесь
        if token:
            print("  [сравнение] Тест в прокси отправляет token (в api_key как giga-auth-<token>): ", _auth_fingerprint(token))
        base_url = get_gpt2giga_base_url().rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        client = AsyncOpenAI(base_url=base_url, api_key=api_key_gpt2giga)
        print("Отправка запроса «Как дела» в GigaChat через gpt2giga...")

    response = await client.chat.completions.create(
        model="GigaChat",
        messages=[{"role": "user", "content": "Как дела"}],
        max_tokens=500,
    )
    content = response.choices[0].message.content if response.choices else ""
    print("Ответ GigaChat:")
    print(content or "(пусто)")


if __name__ == "__main__":
    asyncio.run(main())
