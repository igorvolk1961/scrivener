"""
Управление локальным хранилищем для эмулятора КФО.
"""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

STORAGE_DIR = Path("data/debug_storage")
CHAT_HISTORY_FILENAME = "chat_history.json"
USER_INFO_FILENAME = "user_info.json"


def ensure_storage_dir():
    """Создает директорию хранилища, если она не существует."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_irv_dir(irv_id: str) -> Path:
    """Возвращает путь к директории ИО."""
    return STORAGE_DIR / irv_id


def get_files_dir(irv_id: str) -> Path:
    """Возвращает путь к директории файлов ИО."""
    return get_irv_dir(irv_id) / "files"


def load_metadata(irv_id: str) -> Optional[Dict[str, Any]]:
    """Загружает метаданные ИО."""
    metadata_path = get_irv_dir(irv_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки метаданных для {irv_id}: {e}")
        return None


def save_metadata(irv_id: str, metadata: Dict[str, Any]):
    """Сохраняет метаданные ИО."""
    ensure_storage_dir()
    irv_dir = get_irv_dir(irv_id)
    irv_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = irv_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def get_irv(irv_id: str, **params) -> Optional[Dict[str, Any]]:
    """
    Получает данные ИО.
    
    Args:
        irv_id: UUID версии ИО
        **params: Параметры запроса (with_meta, with_files, и т.д.)
    
    Returns:
        Словарь с данными ИО или None если не найден
    """
    metadata = load_metadata(irv_id)
    if not metadata:
        return None
    
    result = metadata.copy()
    
    # Если нужны файлы, добавляем их список
    if params.get("with_files", True):
        files = get_irv_files(irv_id)
        result["files"] = files
    
    return result


def get_irv_files(irv_id: str) -> List[Dict[str, Any]]:
    """Возвращает список файлов ИО."""
    files_dir = get_files_dir(irv_id)
    if not files_dir.exists():
        return []
    
    files = []
    for file_path in files_dir.iterdir():
        if file_path.is_file() and file_path.name != "metadata.json":
            # Извлекаем irvf_id и имя файла из имени файла
            # Формат: {irvf_id}_{filename}
            parts = file_path.name.split("_", 1)
            if len(parts) == 2:
                irvf_id, filename = parts
            else:
                # Если формат не соответствует, генерируем UUID
                irvf_id = str(uuid.uuid4())
                filename = file_path.name
            
            file_size = file_path.stat().st_size
            files.append({
                "irvfId": irvf_id,
                "name": filename,
                "size": file_size,
                "fileName": filename,
            })
    
    return files


def get_file_content(irvf_id: str) -> Optional[bytes]:
    """
    Читает содержимое файла по irvf_id.
    
    Args:
        irvf_id: UUID версии файла
    
    Returns:
        Содержимое файла в байтах или None если не найден
    """
    # Ищем файл во всех ИО
    for irv_dir in STORAGE_DIR.iterdir():
        if not irv_dir.is_dir():
            continue
        
        files_dir = irv_dir / "files"
        if not files_dir.exists():
            continue
        
        for file_path in files_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith(f"{irvf_id}_"):
                return file_path.read_bytes()
    
    return None


def save_file(irvf_id: str, filename: str, content: bytes, irv_id: Optional[str] = None) -> str:
    """
    Сохраняет файл.
    
    Args:
        irvf_id: UUID версии файла
        filename: Имя файла
        content: Содержимое файла
        irv_id: UUID версии ИО (если None, ищем существующий ИО с таким файлом или создаем новый)
    
    Returns:
        UUID версии ИО
    """
    ensure_storage_dir()
    
    # Если irv_id не указан, ищем существующий ИО с таким файлом
    if not irv_id:
        for existing_irv_dir in STORAGE_DIR.iterdir():
            if not existing_irv_dir.is_dir():
                continue
            existing_irv_id = existing_irv_dir.name
            files_dir = existing_irv_dir / "files"
            if files_dir.exists():
                for file_path in files_dir.iterdir():
                    if file_path.is_file() and file_path.name.startswith(f"{irvf_id}_"):
                        irv_id = existing_irv_id
                        break
            if irv_id:
                break
    
    # Если все еще не найден, создаем новый ИО
    if not irv_id:
        irv_id = str(uuid.uuid4())
        metadata = {
            "irv_id": irv_id,
            "name": filename,
            "description": f"Файл {filename}",
            "ir": {
                "id": str(uuid.uuid4()),
                "parentId": None,
                "nauId": None,
            },
            "files": []
        }
        save_metadata(irv_id, metadata)
    
    # Сохраняем файл
    files_dir = get_files_dir(irv_id)
    files_dir.mkdir(parents=True, exist_ok=True)
    file_path = files_dir / f"{irvf_id}_{filename}"
    file_path.write_bytes(content)
    
    # Обновляем метаданные
    metadata = load_metadata(irv_id)
    if metadata:
        # Проверяем, есть ли уже такой файл в списке
        files = metadata.get("files", [])
        file_exists = any(f.get("irvfId") == irvf_id for f in files)
        if not file_exists:
            files.append({
                "irvfId": irvf_id,
                "name": filename,
                "size": len(content),
                "fileName": filename,
            })
            metadata["files"] = files
            save_metadata(irv_id, metadata)
    
    return irv_id


def create_irv(irv_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создает новый ИО.
    
    Args:
        irv_data: Данные для создания ИО
    
    Returns:
        Созданный ИО с irv_id
    """
    ensure_storage_dir()
    
    irv_id = str(uuid.uuid4())
    io_id = irv_data.get("ioId") or str(uuid.uuid4())
    
    metadata = {
        "irv_id": irv_id,
        "name": irv_data.get("name", "Новый ИО"),
        "description": irv_data.get("description", ""),
        "ir": {
            "id": io_id,
            "parentId": irv_data.get("parentId"),
            "nauId": irv_data.get("nauId"),
        },
        "files": []
    }
    
    # Если указаны файлы, создаем их
    if "fileName" in irv_data:
        file_names = irv_data["fileName"].split(irv_data.get("fileNameSeparator", ","))
        for file_name in file_names:
            file_name = file_name.strip()
            if file_name:
                irvf_id = str(uuid.uuid4())
                metadata["files"].append({
                    "irvfId": irvf_id,
                    "name": file_name,
                    "size": 0,
                    "fileName": file_name,
                })
    
    save_metadata(irv_id, metadata)
    
    return {
        "irv_id": irv_id,
        "ir": {
            "id": io_id,
            "parentId": metadata["ir"]["parentId"],
            "nauId": metadata["ir"]["nauId"],
        },
        "name": metadata["name"],
        "description": metadata["description"],
    }


def update_irv_metadata(irv_id: str, metadata_updates: Dict[str, Any]):
    """Обновляет метаданные ИО."""
    metadata = load_metadata(irv_id)
    if metadata:
        metadata.update(metadata_updates)
        save_metadata(irv_id, metadata)


def delete_irv(irv_id: str):
    """Удаляет ИО и все его файлы."""
    irv_dir = get_irv_dir(irv_id)
    if irv_dir.exists():
        shutil.rmtree(irv_dir)


def get_user_info() -> Dict[str, Any]:
    """
    Загружает информацию о текущем пользователе.
    
    Returns:
        Словарь с информацией о пользователе (по умолчанию возвращает дефолтные значения)
    """
    ensure_storage_dir()
    user_info_path = STORAGE_DIR / USER_INFO_FILENAME
    if not user_info_path.exists():
        # Возвращаем дефолтные значения
        return {
            "id": "debug_user",
            "name": "Тестовый Пользователь",
            "email": "test@example.com",
            "userPost": "Тестовый пользователь",
        }
    try:
        with open(user_info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки информации о пользователе: {e}")
        return {
            "id": "debug_user",
            "name": "Тестовый Пользователь",
            "email": "test@example.com",
            "userPost": "Тестовый пользователь",
        }


def save_user_info(user_info: Dict[str, Any]):
    """
    Сохраняет информацию о текущем пользователе.
    
    Args:
        user_info: Словарь с информацией о пользователе
    """
    ensure_storage_dir()
    user_info_path = STORAGE_DIR / USER_INFO_FILENAME
    with open(user_info_path, "w", encoding="utf-8") as f:
        json.dump(user_info, f, ensure_ascii=False, indent=2)
    logger.info("Информация о пользователе сохранена")

