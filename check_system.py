"""
Скрипт для проверки готовности системы к работе.

Проверяет:
1. Установленные зависимости
2. Подключение к Qdrant
3. Конфигурационные файлы
4. RAG компоненты
"""

import sys
import logging
from pathlib import Path

# Настройка базового логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Проверка версии Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Требуется Python 3.8+, текущая версия: {version.major}.{version.minor}")
        return False
    logger.info(f"✓ Python версия: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Проверка установленных зависимостей."""
    required_packages = [
        'llama_index',
        'qdrant_client',
        'httpx',
        'yaml',
        'pydantic',
        'loguru'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                __import__('yaml')
            elif package == 'llama_index':
                __import__('llama_index')
            else:
                __import__(package)
            logger.info(f"✓ Пакет {package} установлен")
        except ImportError:
            logger.error(f"✗ Пакет {package} не установлен")
            missing.append(package)
    
    # Проверка SmartChanker
    try:
        from smart_chanker.smart_chanker import SmartChanker
        logger.info("✓ SmartChanker установлен")
    except ImportError:
        logger.error("✗ SmartChanker не установлен. Установите: pip install git+https://github.com/igorvolk1961/smart_chanker.git")
        missing.append('smart_chanker')
    
    if missing:
        logger.error(f"\nНе установлены пакеты: {', '.join(missing)}")
        logger.error("Установите их командой: pip install -r requirements.txt")
        return False
    
    return True


def check_qdrant():
    """Проверка подключения к Qdrant."""
    try:
        import httpx
        
        # Проверяем корневой endpoint (работает в версии 1.15.5)
        response = httpx.get("http://localhost:6333/", timeout=5)
        if response.status_code == 200:
            try:
                info = response.json()
                version = info.get("version", "unknown")
                logger.info(f"✓ Qdrant доступен на http://localhost:6333 (версия: {version})")
            except:
                logger.info("✓ Qdrant доступен на http://localhost:6333")
            
            # Дополнительная проверка через API коллекций
            try:
                collections_response = httpx.get("http://localhost:6333/collections", timeout=5)
                if collections_response.status_code == 200:
                    logger.info("✓ Qdrant API работает корректно")
                return True
            except:
                # Если коллекции недоступны, но корневой endpoint работает - это нормально
                return True
        else:
            logger.error(f"✗ Qdrant вернул статус {response.status_code}")
            return False
    except httpx.ConnectError:
        logger.error("✗ Не удалось подключиться к Qdrant на http://localhost:6333")
        logger.error("  Убедитесь, что Qdrant запущен")
        return False
    except Exception as e:
        logger.error(f"✗ Ошибка при проверке Qdrant: {e}")
        return False


def check_config_files():
    """Проверка наличия конфигурационных файлов."""
    config_files = {
        'config.yaml': 'Основная конфигурация RAG-системы',
        'config.json': 'Конфигурация SmartChanker'
    }
    
    all_ok = True
    for file_path, description in config_files.items():
        if Path(file_path).exists():
            logger.info(f"✓ {file_path} найден ({description})")
        else:
            logger.error(f"✗ {file_path} не найден ({description})")
            all_ok = False
    
    # Проверка .env (опциональный)
    if Path('.env').exists():
        logger.info("✓ .env файл найден")
    else:
        logger.warning("⚠ .env файл не найден (опционально, можно создать из .env.example)")
    
    return all_ok


def check_rag_pipeline():
    """Проверка инициализации RAG-пайплайна."""
    try:
        from utils.config import get_config
        from rag.vector_store import QdrantVectorStoreManager
        from rag.chunker_integration import ChunkerIntegration
        
        logger.info("Проверка инициализации компонентов RAG...")
        
        config = get_config()
        logger.info("✓ Конфигурация загружена")
        
        # Проверка компонентов (без полной инициализации)
        qdrant_config = config.get("qdrant", {})
        chunker_config = config.get("chunker", {})
        
        logger.info("✓ Конфигурационные секции найдены")
        
        try:
            vector_store = QdrantVectorStoreManager(
                url=qdrant_config.get("url", "http://localhost:6333"),
                collection_name=qdrant_config.get("collection_name", "scrivener_documents"),
                vector_size=qdrant_config.get("vector_size", 1024)
            )
            logger.info("✓ QdrantVectorStoreManager создан")
        except Exception as e:
            logger.error(f"✗ Ошибка при создании QdrantVectorStoreManager: {e}")
            return False
        
        try:
            chunker = ChunkerIntegration(
                chunker_config_path=chunker_config.get("config_path", "config.json")
            )
            logger.info("✓ ChunkerIntegration создан")
        except Exception as e:
            logger.error(f"✗ Ошибка при создании ChunkerIntegration: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Ошибка при проверке RAG-пайплайна: {e}", exc_info=True)
        return False


def main():
    """Основная функция проверки."""
    logger.info("=" * 60)
    logger.info("Проверка готовности системы Smart RAG")
    logger.info("=" * 60)
    logger.info("")
    
    checks = [
        ("Версия Python", check_python_version),
        ("Зависимости Python", check_dependencies),
        ("Конфигурационные файлы", check_config_files),
        ("Подключение к Qdrant", check_qdrant),
        ("RAG компоненты", check_rag_pipeline),
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n[{name}]")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Ошибка при проверке {name}: {e}", exc_info=True)
            results.append((name, False))
    
    # Итоговый отчет
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЕТ")
    logger.info("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ ПРОЙДЕНО" if result else "✗ НЕ ПРОЙДЕНО"
        logger.info(f"{name}: {status}")
        if not result:
            all_passed = False
    
    logger.info("")
    if all_passed:
        logger.info("✓ Все проверки пройдены! Система готова к работе.")
        logger.info("\nЗапустите пример использования:")
        logger.info("  python example_usage.py")
        return 0
    else:
        logger.error("✗ Некоторые проверки не пройдены. Исправьте ошибки и повторите проверку.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

