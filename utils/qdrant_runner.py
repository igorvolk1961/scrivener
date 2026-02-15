"""
Автозапуск локального Qdrant при старте приложения.

Проверяет доступность Qdrant по url; если не доступен и в конфиге указаны
executable_path и config_path — запускает локальный процесс Qdrant.
Хранилище задаётся через storage_path (передаётся как QDRANT__STORAGE__STORAGE_PATH).
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from loguru import logger


def _qdrant_health_check(url: str, timeout: float = 2.0) -> bool:
    """Проверка доступности Qdrant по URL (GET /)."""
    try:
        req = Request(url.rstrip("/") + "/", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _is_local_url(url: str) -> bool:
    """Проверка, что URL указывает на локальный Qdrant (можно запускать локально)."""
    if not url:
        return False
    url_lower = url.lower().strip()
    return (
        url_lower.startswith("http://localhost:")
        or url_lower.startswith("http://127.0.0.1:")
    )


def ensure_qdrant_started(
    qdrant_config: dict,
    *,
    health_url: Optional[str] = None,
    wait_timeout: float = 30.0,
    check_interval: float = 1.0,
) -> bool:
    """
    Убедиться, что Qdrant доступен: если нет — попытаться запустить локальный процесс.

    Использует из qdrant_config:
    - url: URL API Qdrant (для проверки и только для localhost — автозапуск)
    - auto_start: разрешить автозапуск (по умолчанию True, если задан executable_path)
    - storage_path: папка хранилища (относительно cwd или абсолютный путь); передаётся в Qdrant
    - executable_path: путь к qdrant.exe (или бинарнику)
    - config_path: путь к конфигу Qdrant (передаётся как --config-path)

    Returns:
        True, если Qdrant доступен (уже был или успешно запущен); False иначе.
    """
    url = (health_url or qdrant_config.get("url") or "http://localhost:6333").strip().rstrip("/")
    base_url = url

    # Уже доступен
    if _qdrant_health_check(base_url):
        logger.debug("Qdrant уже доступен: {}", base_url)
        return True

    # Не локальный URL — не запускаем процесс
    if not _is_local_url(base_url):
        logger.debug("Qdrant URL не локальный, автозапуск не выполняется: {}", base_url)
        return False

    auto_start = qdrant_config.get("auto_start", True)
    executable_path = qdrant_config.get("executable_path") or ""
    config_path = qdrant_config.get("config_path") or ""

    executable_path = executable_path.strip()
    config_path = config_path.strip()

    if not auto_start or not executable_path:
        logger.debug(
            "Автозапуск Qdrant отключён (auto_start={}, executable_path задан={})",
            auto_start,
            bool(executable_path),
        )
        return False

    exe = Path(executable_path)
    if not exe.exists():
        logger.warning(
            "Исполняемый файл Qdrant не найден: {}. Запустите Qdrant вручную.",
            executable_path,
        )
        return False

    cmd = [str(exe)]
    if config_path:
        cfg = Path(config_path)
        if cfg.exists():
            cmd.extend(["--config-path", str(cfg)])
        else:
            logger.warning("Файл конфигурации Qdrant не найден: {}", config_path)

    # Путь хранилища: относительный — от текущей рабочей директории (корень проекта при запуске из main.py)
    storage_path = (qdrant_config.get("storage_path") or "data/qdrant_storage").strip()
    storage_dir = Path(storage_path)
    if not storage_dir.is_absolute():
        storage_dir = Path.cwd() / storage_dir
    storage_dir = storage_dir.resolve()
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Не удалось создать папку хранилища Qdrant {}: {}", storage_dir, e)
        return False
    env = {**os.environ, "QDRANT__STORAGE__STORAGE_PATH": str(storage_dir)}

    try:
        logger.info("Запуск локального Qdrant (хранилище: {}): {}", storage_dir, " ".join(cmd))
        # Запуск в фоне, без привязки к консоли (CREATE_NO_WINDOW на Windows)
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW  # 0x08000000

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(exe.parent),
            env=env,
            creationflags=creationflags,
        )
    except Exception as e:
        logger.error("Не удалось запустить Qdrant: {}", e)
        return False

    # Ждём появления сервиса
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        time.sleep(check_interval)
        if _qdrant_health_check(base_url):
            logger.info("Qdrant успешно запущен и доступен: {}", base_url)
            return True
        if proc.poll() is not None:
            logger.error("Процесс Qdrant завершился с кодом {}", proc.returncode)
            return False

    logger.warning(
        "Qdrant не ответил в течение {} с. Проверьте {} вручную.",
        wait_timeout,
        base_url,
    )
    return False
