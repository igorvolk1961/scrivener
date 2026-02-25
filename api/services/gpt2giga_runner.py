"""
Автозапуск прокси gpt2giga при использовании GigaChat на том же сервере.

Если GigaChatAuthProvider используется с URL на localhost (например http://localhost:8090),
проверяется доступность сервиса и при необходимости запускается процесс gpt2giga
(команда `gpt2giga` из PATH или `python -m gpt2giga`).
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

import httpx
from loguru import logger

# Порт по умолчанию (совпадает с gpt2giga)
_DEFAULT_PORT = 8090
# Таймаут проверки доступности (секунды)
_CHECK_TIMEOUT = 3.0
# Ожидание после запуска процесса перед проверкой (секунды)
_STARTUP_WAIT = 2.0
# Максимум попыток проверки после запуска
_STARTUP_RETRIES = 15

_lock = threading.Lock()
_started_urls: set[str] = set()


def get_gpt2giga_base_url() -> str:
    """URL прокси gpt2giga: из окружения или localhost по умолчанию."""
    url = os.environ.get("GPT2GIGA_BASE_URL", "").strip()
    if not url:
        return f"http://localhost:{_DEFAULT_PORT}"
    return url.rstrip("/")


def _is_local_host(host: str) -> bool:
    if not host:
        return True
    h = host.lower()
    return h in ("localhost", "127.0.0.1", "::1")


def _parse_url(url: str) -> tuple[str, int]:
    """Возвращает (host, port) из URL."""
    parsed = urlparse(url)
    host = (parsed.hostname or "localhost").lower()
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
        if parsed.scheme != "https" and "localhost" in host or host == "127.0.0.1":
            port = _DEFAULT_PORT
    return host, port


def _check_running(base_url: str) -> bool:
    """Проверяет, отвечает ли прокси по base_url (GET /v1/models или /docs)."""
    try:
        with httpx.Client(timeout=_CHECK_TIMEOUT, verify=False) as client:
            # OpenAI-совместимый эндпоинт или FastAPI docs
            for path in ("/v1/models", "/docs"):
                r = client.get(f"{base_url.rstrip('/')}{path}")
                if r.status_code in (200, 401):
                    return True
    except Exception:
        pass
    return False


def _start_process(port: int) -> subprocess.Popen | None:
    """Запускает gpt2giga в фоне. Возвращает Popen или None при ошибке."""
    # Аргументы как при ручном запуске: передаём модель из запроса клиента, токен и отключаем проверку SSL
    base_args = [
        "--proxy.port", str(port),
        "--proxy.pass-token", "true",
        "--proxy.pass-model", "true",
        "--gigachat.verify-ssl-certs", "false",
    ]
    cmd = shutil.which("gpt2giga")
    if cmd:
        argv = [cmd] + base_args
    else:
        try:
            import gpt2giga  # noqa: F401
        except ImportError:
            logger.warning(
                "gpt2giga не установлен: установите через pip install gpt2giga или uv tool install gpt2giga"
            )
            return None
        argv = [sys.executable, "-m", "gpt2giga"] + base_args
    env = os.environ.copy()
    env["GPT2GIGA_PORT"] = str(port)
    try:
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        logger.info(f"Запущен процесс gpt2giga (PID {proc.pid}, порт {port})")
        return proc
    except Exception as e:
        logger.warning(f"Не удалось запустить gpt2giga: {e}")
        return None


def ensure_gpt2giga_running(base_url: str | None = None) -> None:
    """
    Убеждается, что прокси gpt2giga запущен по указанному URL.

    Если base_url на localhost и сервис не отвечает — запускает процесс gpt2giga.
    Для удалённого URL только логирует (не запускает процесс).
    Вызов потокобезопасен.
    """
    url = (base_url or get_gpt2giga_base_url()).rstrip("/")
    host, port = _parse_url(url)
    if not _is_local_host(host):
        return
    with _lock:
        if url in _started_urls:
            return
        if _check_running(url):
            _started_urls.add(url)
            return
        proc = _start_process(port)
        if proc is None:
            return
        _started_urls.add(url)
    # Ждём готовности без держания блокировки
    for _ in range(_STARTUP_RETRIES):
        time.sleep(_STARTUP_WAIT)
        if _check_running(url):
            return
    logger.warning(
        f"gpt2giga запущен (PID {proc.pid}), но не ответил по {url} за {_STARTUP_RETRIES * _STARTUP_WAIT} с"
    )
