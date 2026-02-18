#!/usr/bin/env python3
"""
Утилита для проверки наличия чанков с указанным section_number в базе данных Qdrant.

Использование:
    # С активированным виртуальным окружением:
    python utils/check_section_number.py "2.1.2.3.T7"
    python utils/check_section_number.py "2.1.2.3.T7" --vdb-url http://localhost:6333
    python utils/check_section_number.py "2.1.2.3.T7" --collection scrivener_documents
    python utils/check_section_number.py "2.1.2.3.T7" --json  # JSON формат
    python utils/check_section_number.py "2.1.2.3.T7" --verbose  # Подробный вывод
"""

import argparse
import json
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

# Получаем пути
script_dir = Path(__file__).parent
project_root = script_dir.parent
rag_dir = project_root / "rag"
utils_dir = project_root / "utils"

# Временно удаляем utils из sys.path, чтобы избежать конфликта с utils/logging.py
# при импорте qdrant_client (который использует стандартный logging)
utils_path_str = str(utils_dir)
if utils_path_str in sys.path:
    sys.path.remove(utils_path_str)

# Добавляем project_root в sys.path для импорта rag модулей
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # Импортируем qdrant_client (теперь он будет использовать стандартный logging)
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from rag.vector_store import QdrantVectorStoreManager
    
    # Импортируем config напрямую через importlib, избегая импорта utils.logging
    config_path = utils_dir / "config.py"
    spec = importlib.util.spec_from_file_location("utils_config", config_path)
    utils_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_config)
    get_config = utils_config.get_config
    
except ImportError as e:
    print(f"Ошибка импорта: {e}", file=sys.stderr)
    print("\nУбедитесь, что виртуальное окружение активировано:", file=sys.stderr)
    print("  source .venv/bin/activate  # Linux/Mac", file=sys.stderr)
    print("  .venv\\Scripts\\activate     # Windows", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Ошибка при загрузке модулей: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)


def find_chunks_by_section_number(
    section_number: str,
    vdb_url: str = None,
    collection_name: str = None,
    limit: int = 100,
    use_prefix: bool = False
) -> List[Dict[str, Any]]:
    """
    Поиск чанков по section_number в Qdrant.
    
    Args:
        section_number: Номер раздела для поиска
        vdb_url: URL Qdrant (если не указан, берется из конфигурации)
        collection_name: Имя коллекции (если не указано, берется из конфигурации)
        limit: Максимальное количество результатов
        use_prefix: Если True, ищет чанки с section_number, начинающимся с указанного значения
    
    Returns:
        Список найденных чанков
    """
    # Загружаем конфигурацию
    config = get_config()
    qdrant_config = config.get("qdrant", {})
    
    # Используем параметры из аргументов или конфигурации
    if vdb_url is None:
        vdb_url = qdrant_config.get("url", "http://localhost:6333")
    
    if collection_name is None:
        collection_name = qdrant_config.get("collection_name", "scrivener_documents")
    
    # Нормализуем URL
    vdb_url = vdb_url.strip().rstrip("/")
    if not vdb_url.startswith("http"):
        vdb_url = f"http://{vdb_url}"
    
    print(f"Подключение к Qdrant: {vdb_url}")
    print(f"Коллекция: {collection_name}")
    if use_prefix:
        print(f"Поиск чанков с section_number, начинающимся с: {section_number}")
    else:
        print(f"Поиск чанков с section_number: {section_number}")
    print("-" * 80)
    
    try:
        # Инициализация векторного хранилища
        # Размер вектора будет определен из существующей коллекции или через пробный запрос
        # Для операций чтения можно использовать None, размер определится из коллекции
        vector_store_manager = QdrantVectorStoreManager(
            url=vdb_url,
            api_key=qdrant_config.get("api_key"),
            collection_name=collection_name,
            vector_size=None,  # Будет определен из существующей коллекции
            timeout=qdrant_config.get("timeout", 30)
        )
        
        if use_prefix:
            # Для поиска по префиксу получаем все чанки и фильтруем на стороне Python
            # (Qdrant не поддерживает поиск по префиксу строк напрямую)
            results = vector_store_manager.client.scroll(
                collection_name=collection_name,
                scroll_filter=None,  # Без фильтра - получаем все
                limit=10000,  # Большой лимит для поиска по префиксу
                with_payload=True,
                with_vectors=False
            )
        else:
            # Создание фильтра для поиска по точному совпадению section_number
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="section_number",
                        match=MatchValue(value=section_number)
                    )
                ]
            )
            
            # Поиск чанков
            results = vector_store_manager.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
        
        # Преобразование результатов
        chunks = []
        for point in results[0]:
            payload = point.payload if point.payload else {}
            
            # Для поиска по префиксу фильтруем на стороне Python
            if use_prefix:
                chunk_section_number = payload.get("section_number", "")
                if not chunk_section_number or not str(chunk_section_number).startswith(section_number):
                    continue
            
            # Собираем все метаданные из payload
            chunk_data = {
                "id": str(point.id),
                "text": payload.get("text", "")[:200] + "..." if len(payload.get("text", "")) > 200 else payload.get("text", ""),
                "metadata": {
                    "section_number": payload.get("section_number"),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_type": payload.get("chunk_type"),
                    "chunk_number": payload.get("chunk_number"),
                    "irv_id": payload.get("irv_id"),
                    "irvf_id": payload.get("irvf_id"),
                    "document_id": payload.get("document_id"),
                    "file_name": payload.get("file_name"),
                    "table_id": payload.get("table_id"),
                    "table_name": payload.get("table_name"),
                },
                # Полный payload для отладки
                "full_payload_keys": list(payload.keys()) if payload else []
            }
            chunks.append(chunk_data)
        
        return chunks
        
    except Exception as e:
        print(f"Ошибка при поиске чанков: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return []


def list_all_section_numbers(
    vdb_url: str = None,
    collection_name: str = None,
    limit: int = 10000
):
    """
    Выводит все уникальные section_number в базе данных.
    
    Args:
        vdb_url: URL Qdrant
        collection_name: Имя коллекции
        limit: Максимальное количество чанков для проверки
    """
    config = get_config()
    qdrant_config = config.get("qdrant", {})
    
    if vdb_url is None:
        vdb_url = qdrant_config.get("url", "http://localhost:6333")
    
    if collection_name is None:
        collection_name = qdrant_config.get("collection_name", "scrivener_documents")
    
    vdb_url = vdb_url.strip().rstrip("/")
    if not vdb_url.startswith("http"):
        vdb_url = f"http://{vdb_url}"
    
    print(f"Подключение к Qdrant: {vdb_url}")
    print(f"Коллекция: {collection_name}")
    print("Поиск всех уникальных section_number...")
    print("-" * 80)
    
    try:
        # Размер вектора будет определен из существующей коллекции или через пробный запрос
        # Для операций чтения можно использовать None, размер определится из коллекции
        vector_store_manager = QdrantVectorStoreManager(
            url=vdb_url,
            api_key=qdrant_config.get("api_key"),
            collection_name=collection_name,
            vector_size=None,  # Будет определен из существующей коллекции
            timeout=qdrant_config.get("timeout", 30)
        )
        
        # Получаем все чанки
        results = vector_store_manager.client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Собираем уникальные section_number
        section_numbers = set()
        chunks_without_section = 0
        total_chunks = 0
        
        for point in results[0]:
            total_chunks += 1
            payload = point.payload if point.payload else {}
            section_number = payload.get("section_number")
            
            if section_number:
                section_numbers.add(str(section_number))
            else:
                chunks_without_section += 1
        
        print(f"\nВсего чанков проверено: {total_chunks}")
        print(f"Чанков без section_number: {chunks_without_section}")
        print(f"Уникальных section_number: {len(section_numbers)}")
        print("\nСписок всех section_number:")
        print("=" * 80)
        
        # Сортируем для удобства
        sorted_sections = sorted(section_numbers)
        for i, section in enumerate(sorted_sections, 1):
            print(f"{i:4d}. {section}")
        
        # Проверяем, есть ли похожие на искомый
        search_value = "2.1.2.3.T7"
        similar = [s for s in sorted_sections if search_value in s or s in search_value]
        if similar:
            print(f"\nПохожие на '{search_value}':")
            for s in similar:
                print(f"  - {s}")
        
        # Проверяем наличие чанков таблиц
        print("\n" + "=" * 80)
        print("Проверка наличия чанков таблиц...")
        table_chunks_count = 0
        table_sections = set()
        
        for point in results[0]:
            payload = point.payload if point.payload else {}
            chunk_type = payload.get("chunk_type")
            if chunk_type == "table":
                table_chunks_count += 1
                section = payload.get("section_number")
                if section:
                    table_sections.add(str(section))
        
        print(f"Чанков с chunk_type='table': {table_chunks_count}")
        if table_sections:
            print(f"Уникальных section_number для таблиц: {len(table_sections)}")
            print("Section_number для таблиц:")
            for s in sorted(table_sections):
                print(f"  - {s}")
        
        # Проверяем, есть ли чанки с table_id
        print("\n" + "=" * 80)
        print("Проверка наличия table_id в метаданных...")
        chunks_with_table_id = 0
        table_ids = set()
        
        for point in results[0]:
            payload = point.payload if point.payload else {}
            table_id = payload.get("table_id")
            if table_id:
                chunks_with_table_id += 1
                table_ids.add(str(table_id))
        
        print(f"Чанков с table_id: {chunks_with_table_id}")
        if table_ids:
            print(f"Уникальных table_id: {len(table_ids)}")
            print("Первые 20 table_id:")
            for i, tid in enumerate(sorted(table_ids)[:20], 1):
                print(f"  {i}. {tid}")
        
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


def main():
    """Главная функция утилиты."""
    parser = argparse.ArgumentParser(
        description="Поиск чанков по section_number в Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "section_number",
        type=str,
        nargs="?",
        default=None,
        help="Номер раздела для поиска (например, '2.1.2.3.T7'). Не требуется при использовании --list-all"
    )
    
    parser.add_argument(
        "--vdb-url",
        type=str,
        default=None,
        help="URL Qdrant (по умолчанию из config.yaml)"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Имя коллекции (по умолчанию из config.yaml)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Максимальное количество результатов (по умолчанию: 100)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести результаты в формате JSON"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="Искать чанки с section_number, начинающимся с указанного значения"
    )
    
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="Показать все уникальные section_number в базе данных"
    )
    
    args = parser.parse_args()
    
    # Если запрошен список всех section_number
    if args.list_all:
        list_all_section_numbers(
            vdb_url=args.vdb_url,
            collection_name=args.collection,
            limit=args.limit
        )
        sys.exit(0)
    
    # Проверяем, что section_number указан
    if not args.section_number:
        parser.error("section_number обязателен, если не используется --list-all")
    
    # Поиск чанков
    chunks = find_chunks_by_section_number(
        section_number=args.section_number,
        vdb_url=args.vdb_url,
        collection_name=args.collection,
        limit=args.limit,
        use_prefix=args.prefix
    )
    
    # Вывод результатов
    if args.json:
        # JSON формат
        output = {
            "section_number": args.section_number,
            "found": len(chunks),
            "chunks": chunks
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # Текстовый формат
        print(f"\nНайдено чанков: {len(chunks)}")
        print("=" * 80)
        
        if not chunks:
            print("Чанки с указанным section_number не найдены.")
        else:
            for i, chunk in enumerate(chunks, 1):
                print(f"\nЧанк #{i}:")
                print(f"  ID: {chunk['id']}")
                print(f"  Section Number: {chunk['metadata'].get('section_number')}")
                print(f"  Chunk Index: {chunk['metadata'].get('chunk_index')}")
                print(f"  Chunk Type: {chunk['metadata'].get('chunk_type')}")
                print(f"  IRV ID: {chunk['metadata'].get('irv_id')}")
                print(f"  IRVF ID: {chunk['metadata'].get('irvf_id')}")
                print(f"  Document ID: {chunk['metadata'].get('document_id')}")
                print(f"  File Name: {chunk['metadata'].get('file_name')}")
                print(f"  Chunk Number: {chunk['metadata'].get('chunk_number')}")
                print(f"  Table ID: {chunk['metadata'].get('table_id')}")
                print(f"  Table Name: {chunk['metadata'].get('table_name')}")
                
                if args.verbose:
                    print(f"  Все ключи в payload: {chunk.get('full_payload_keys', [])}")
                    text = chunk.get('text', '')
                    if len(text) > 500:
                        print(f"  Text (первые 500 символов):")
                        print(f"    {text[:500]}...")
                    else:
                        print(f"  Text:")
                        print(f"    {text}")
                else:
                    text = chunk.get('text', '')
                    if text:
                        print(f"  Text (первые 200 символов): {text[:200]}...")
                
                print("-" * 80)
    
    # Код возврата
    sys.exit(0 if chunks else 1)


if __name__ == "__main__":
    main()

