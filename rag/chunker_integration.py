"""
Интеграция SmartChanker для обработки документов.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from smart_chanker.smart_chanker import SmartChanker
except ImportError:
    raise ImportError(
        "SmartChanker не установлен. Установите его командой: "
        "pip install git+https://github.com/igorvolk1961/smart_chanker.git"
    )

logger = logging.getLogger(__name__)


class ChunkerIntegration:
    """
    Класс для интеграции SmartChanker в RAG-систему.
    
    Обрабатывает документы через SmartChanker и подготавливает
    чанки с метаданными для индексации в векторное хранилище.
    """
    
    def __init__(self, chunker_config_path: str, output_dir: Optional[str] = None):
        """
        Инициализация интеграции с SmartChanker.
        
        Args:
            chunker_config_path: Путь к конфигурационному файлу SmartChanker
            output_dir: Директория для сохранения результатов чанкинга (если None, берется из конфига чанкера)
        """
        self.chunker_config_path = Path(chunker_config_path)
        
        # Инициализация SmartChanker
        if not self.chunker_config_path.exists():
            logger.warning(
                f"Конфигурационный файл SmartChanker не найден: {chunker_config_path}. "
                "Создайте файл config.json с настройками чанкера."
            )
        
        self.chunker = SmartChanker(str(self.chunker_config_path))
        logger.info(f"SmartChanker инициализирован с конфигом: {chunker_config_path}")
        
        # Загружаем настройки из конфига чанкера
        self.save_result_json = False
        default_output_dir = "data/chunks"
        
        try:
            if self.chunker_config_path.exists():
                with open(self.chunker_config_path, 'r', encoding='utf-8') as f:
                    chunker_config_data = json.load(f)
                    output_config = chunker_config_data.get("output", {})
                    self.save_result_json = output_config.get("save_result_json", False)
                    # output_dir из конфига чанкера имеет приоритет над переданным параметром
                    config_output_dir = output_config.get("output_dir")
                    if config_output_dir:
                        default_output_dir = config_output_dir
                    if self.save_result_json:
                        logger.info("Сохранение результатов чанкера в JSON файл включено (из конфига чанкера)")
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфиг чанкера: {e}")
        
        # Используем переданный output_dir или из конфига
        self.output_dir = Path(output_dir if output_dir is not None else default_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Директория для результатов чанкера: {self.output_dir}")
    
    def process_document(
        self, 
        document_path: str, 
        document_id: Optional[str] = None,
        max_chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Обработка документа через SmartChanker.
        
        Args:
            document_path: Путь к документу для обработки
            document_id: Уникальный идентификатор документа (если None, генерируется из имени файла)
            max_chunk_size: Максимальный размер чанка в символах (если указан, переопределяет значение из конфига)
        
        Returns:
            Словарь с результатами обработки:
            - chunks: список чанков с метаданными
            - metadata: общие метаданные документа
            - toc_chunks: чанки оглавления (если есть)
        """
        doc_path = Path(document_path)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Документ не найден: {document_path}")
        
        if document_id is None:
            document_id = doc_path.stem
        
        logger.info(f"Обработка документа: {document_path} (ID: {document_id})")
        
        # Применяем max_chunk_size, если он передан
        original_max_chunk_size = None
        if max_chunk_size is not None:
            # Сохраняем оригинальное значение для восстановления
            if hasattr(self.chunker, 'config') and isinstance(self.chunker.config, dict):
                hierarchical_chunking = self.chunker.config.get("hierarchical_chunking", {})
                if isinstance(hierarchical_chunking, dict):
                    original_max_chunk_size = hierarchical_chunking.get("max_chunk_size")
                    hierarchical_chunking["max_chunk_size"] = max_chunk_size
                    logger.info(f"Используется max_chunk_size из запроса: {max_chunk_size} символов")
                # Также обновляем для table_processing
                table_processing = self.chunker.config.get("table_processing", {})
                if isinstance(table_processing, dict):
                    table_processing["max_chunk_size"] = max_chunk_size
        
        # Создание выходной директории для этого документа
        doc_output_dir = self.output_dir / document_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Обработка документа через SmartChanker
        try:
            result = self.chunker.run_end_to_end(
                str(doc_path),
                str(doc_output_dir)
            )
            
            # Восстанавливаем оригинальное значение max_chunk_size
            if max_chunk_size is not None and original_max_chunk_size is not None:
                if hasattr(self.chunker, 'config') and isinstance(self.chunker.config, dict):
                    hierarchical_chunking = self.chunker.config.get("hierarchical_chunking", {})
                    if isinstance(hierarchical_chunking, dict):
                        hierarchical_chunking["max_chunk_size"] = original_max_chunk_size
                    table_processing = self.chunker.config.get("table_processing", {})
                    if isinstance(table_processing, dict):
                        table_processing["max_chunk_size"] = original_max_chunk_size
            
            logger.info(f"Документ обработан успешно. Результаты сохранены в: {doc_output_dir}")
            
            # Проверяем, что вернул SmartChanker
            if result is not None:
                logger.debug(f"SmartChanker вернул результат типа: {type(result)}")
                if isinstance(result, dict):
                    logger.debug(f"Ключи в результате: {result.keys()}")
            
            # Загрузка и парсинг результатов
            # Сначала пробуем использовать результат напрямую, если он есть
            if result is not None and isinstance(result, dict):
                chunks_data = self._load_chunks_from_dict(result, document_id)
            else:
                # Иначе загружаем из файлов
                chunks_data = self._load_chunks_from_result(doc_output_dir, document_id)
            
            # Формируем полный результат для сохранения
            full_result = {
                "document_id": document_id,
                "document_path": str(doc_path),
                "chunks": chunks_data["chunks"],
                "metadata": chunks_data["metadata"],
                "toc_chunks": chunks_data.get("toc_chunks", []),
                "table_chunks": chunks_data.get("table_chunks", []),
                "output_dir": str(doc_output_dir)
            }
            
            # Сохраняем полный результат в JSON файл для удобства просмотра и отладки (если включено в конфигурации)
            if self.save_result_json:
                result_json_path = doc_output_dir / "chunks_result.json"
                try:
                    with open(result_json_path, 'w', encoding='utf-8') as f:
                        json.dump(full_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Результат чанкера сохранен в JSON файл: {result_json_path}")
                except Exception as e:
                    logger.warning(f"Не удалось сохранить результат в JSON файл: {e}")
            
            return full_result
            
        except Exception as e:
            logger.error(f"Ошибка при обработке документа {document_path}: {e}", exc_info=True)
            raise
    
    def _load_chunks_from_dict(
        self,
        result_dict: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Загрузка чанков из словаря результата SmartChanker.
        
        Args:
            result_dict: Словарь с результатами SmartChanker
            document_id: ID документа
        
        Returns:
            Словарь с чанками и метаданными
        """
        chunks = []
        metadata = {
            "document_id": document_id,
            "total_chunks": 0
        }
        toc_chunks = []
        
        # Извлечение чанков из результата
        if isinstance(result_dict, dict):
            # Пробуем разные возможные ключи
            raw_chunks = (
                result_dict.get("chunks") or
                result_dict.get("hierarchical_chunks") or
                result_dict.get("data") or
                []
            )
            
            # Обработка чанков
            for idx, chunk_data in enumerate(raw_chunks):
                # Отладочное логирование для первых чанков
                if idx < 3:
                    logger.info(f"Чанк {idx} от SmartChanker (тип: {type(chunk_data)}):")
                    if isinstance(chunk_data, dict):
                        logger.info(f"  Ключи: {list(chunk_data.keys())}")
                        # Выводим структуру без текста (может быть длинным)
                        chunk_preview = {k: v for k, v in chunk_data.items() if k != "text"}
                        logger.info(f"  Данные (без text): {chunk_preview}")
                    else:
                        logger.info(f"  Данные: {chunk_data}")
                
                chunk = self._process_chunk(chunk_data, idx, document_id)
                if chunk:
                    chunks.append(chunk)
            
            # Извлечение TOC чанков
            if "toc_chunks" in result_dict:
                for idx, toc_chunk_data in enumerate(result_dict["toc_chunks"]):
                    toc_chunk = self._process_chunk(toc_chunk_data, idx, document_id, is_toc=True)
                    if toc_chunk:
                        toc_chunks.append(toc_chunk)
            
            metadata["total_chunks"] = len(chunks)
            metadata["has_toc"] = len(toc_chunks) > 0
        
        return {
            "chunks": chunks,
            "metadata": metadata,
            "toc_chunks": toc_chunks,
            "table_chunks": result_dict.get("table_chunks", []) if isinstance(result_dict, dict) else []
        }
    
    def _load_chunks_from_result(
        self, 
        output_dir: Path, 
        document_id: str
    ) -> Dict[str, Any]:
        """
        Загрузка чанков из результатов SmartChanker.
        
        Args:
            output_dir: Директория с результатами обработки
            document_id: ID документа
        
        Returns:
            Словарь с чанками и метаданными
        """
        chunks = []
        metadata = {
            "document_id": document_id,
            "total_chunks": 0
        }
        toc_chunks = []
        table_chunks = []
        
        # Поиск JSON файла с результатами
        json_files = list(output_dir.glob("*.json"))
        
        # Если JSON файлов нет, пробуем найти текстовые файлы с чанками
        if not json_files:
            logger.warning(f"JSON файлы с результатами не найдены в {output_dir}")
            
            # Пробуем найти текстовые файлы (SmartChanker может создавать .txt файлы)
            txt_files = [f for f in output_dir.glob("*.txt") if not f.name.endswith("_toc.txt")]
            
            if txt_files:
                logger.info(f"Найдены текстовые файлы: {[f.name for f in txt_files]}")
                # Читаем текстовые файлы как чанки
                for idx, txt_file in enumerate(txt_files):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                chunk = {
                                    "text": text,
                                    "metadata": {
                                        "document_id": document_id,
                                        "chunk_index": idx,
                                        "source_file": txt_file.name
                                    }
                                }
                                chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Ошибка при чтении файла {txt_file}: {e}")
                
                metadata["total_chunks"] = len(chunks)
                return {
                    "chunks": chunks,
                    "metadata": metadata,
                    "toc_chunks": toc_chunks
                }
            else:
                logger.warning("Не найдены ни JSON, ни текстовые файлы с чанками")
                return {
                    "chunks": chunks,
                    "metadata": metadata,
                    "toc_chunks": toc_chunks
                }
        
        # Загрузка основного JSON файла (обычно первый найденный)
        main_json_file = json_files[0]
        
        try:
            with open(main_json_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Извлечение чанков из результата
            # Структура зависит от формата вывода SmartChanker
            if isinstance(result_data, dict):
                # Если результат - словарь с ключом chunks или подобным
                if "chunks" in result_data:
                    raw_chunks = result_data["chunks"]
                elif "hierarchical_chunks" in result_data:
                    raw_chunks = result_data["hierarchical_chunks"]
                else:
                    # Пытаемся найти чанки в других ключах
                    raw_chunks = result_data.get("data", [])
                
                # Обработка чанков
                for idx, chunk_data in enumerate(raw_chunks):
                    chunk = self._process_chunk(chunk_data, idx, document_id)
                    if chunk:
                        chunks.append(chunk)
                
                # Извлечение TOC чанков
                if "toc_chunks" in result_data:
                    for idx, toc_chunk_data in enumerate(result_data["toc_chunks"]):
                        toc_chunk = self._process_chunk(toc_chunk_data, idx, document_id, is_toc=True)
                        if toc_chunk:
                            toc_chunks.append(toc_chunk)
                
                # Извлечение table_chunks из результата (если есть)
                if "table_chunks" in result_data:
                    for idx, table_chunk_data in enumerate(result_data["table_chunks"]):
                        table_chunk = self._process_chunk(table_chunk_data, idx, document_id, is_table=True)
                        if table_chunk:
                            table_chunks.append(table_chunk)
                
                # Извлечение метаданных
                metadata.update({
                    "total_chunks": len(chunks),
                    "has_toc": len(toc_chunks) > 0,
                    "has_tables": len(table_chunks) > 0,
                    "source_file": str(main_json_file)
                })
                
            elif isinstance(result_data, list):
                # Если результат - список чанков
                for idx, chunk_data in enumerate(result_data):
                    chunk = self._process_chunk(chunk_data, idx, document_id)
                    if chunk:
                        chunks.append(chunk)
                
                metadata["total_chunks"] = len(chunks)
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке чанков из {main_json_file}: {e}", exc_info=True)
        
        return {
            "chunks": chunks,
            "metadata": metadata,
            "toc_chunks": toc_chunks,
            "table_chunks": table_chunks if 'table_chunks' in locals() else []
        }
    
    def _process_chunk(
        self, 
        chunk_data: Any, 
        index: int, 
        document_id: str,
        is_toc: bool = False,
        is_table: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Обработка одного чанка и извлечение метаданных.
        
        Args:
            chunk_data: Данные чанка из SmartChanker
            index: Индекс чанка
            document_id: ID документа
            is_toc: Флаг, указывающий что это чанк оглавления
        
        Returns:
            Словарь с обработанным чанком и метаданными
        """
        try:
            if isinstance(chunk_data, dict):
                # Извлечение текста
                text = chunk_data.get("text", chunk_data.get("content", ""))
                
                if not text or not text.strip():
                    return None
                
                # Извлечение метаданных о разделах из чанкера
                # Сначала проверяем, есть ли вложенная структура metadata
                chunk_metadata = chunk_data.get("metadata", {})
                if not isinstance(chunk_metadata, dict):
                    chunk_metadata = {}
                
                # Объединяем chunk_data и chunk_metadata для поиска метаданных
                # chunk_data имеет приоритет, но если там нет, ищем в chunk_metadata
                def get_metadata_value(*keys):
                    """Получить значение из chunk_data или chunk_metadata по списку возможных ключей."""
                    for key in keys:
                        value = chunk_data.get(key)
                        if value is not None:
                            return value
                        value = chunk_metadata.get(key)
                        if value is not None:
                            return value
                    return None
                
                # Извлечение метаданных о разделах из чанкера
                # Пробуем извлечь все возможные поля, связанные с разделами
                # Определяем тип чанка из метаданных (источник истины)
                chunk_type = chunk_metadata.get("chunk_type", "text")
                
                metadata = {
                    "document_id": document_id,
                    "chunk_index": index,
                    "is_toc": is_toc,
                    "is_table": is_table,
                    # Метаданные о разделах
                    "chunk_type": chunk_type,
                    "section_number": get_metadata_value("section_number", "number", "section_num"),
                    "chunk_number": get_metadata_value("chunk_number"),
                    "word_count": get_metadata_value("word_count"),
                    "char_count": get_metadata_value("char_count"),
                    "is_complete_section": get_metadata_value("is_complete_section"),
                }
                
                
                for key, value in chunk_data.items():
                    if value is not None:
                        # Добавляем только если это не сложная структура (dict/list) или если это полезная информация
                        if not isinstance(value, (dict, list)) or (isinstance(value, dict) and len(value) < 10):
                            metadata[key] = value
                
                # Удаление None значений
                metadata_before_cleanup = dict(metadata)  # Сохраняем для отладки
                metadata = {k: v for k, v in metadata.items() if v is not None}
                
                # Отладочное логирование: выводим извлеченные метаданные для первых нескольких чанков
                if index < 3:
                    logger.info(f"Чанк {index}: метаданные ДО удаления None: {metadata_before_cleanup}")
                    logger.info(f"Чанк {index}: метаданные ПОСЛЕ удаления None: {metadata}")
                    removed_keys = set(metadata_before_cleanup.keys()) - set(metadata.keys())
                    if removed_keys:
                        logger.info(f"Чанк {index}: удаленные ключи (были None): {removed_keys}")
                
                return {
                    "text": text.strip(),
                    "metadata": metadata
                }
            
            elif isinstance(chunk_data, str):
                # Если чанк - просто строка
                return {
                    "text": chunk_data.strip(),
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": index,
                        "is_toc": is_toc,
                        "is_table": is_table,
                        "chunk_type": "text"  # По умолчанию для строковых чанков
                    }
                }
            
            else:
                logger.warning(f"Неожиданный формат чанка: {type(chunk_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при обработке чанка {index}: {e}", exc_info=True)
            return None
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Обработка всех документов в папке.
        
        Args:
            folder_path: Путь к папке с документами
        
        Returns:
            Список результатов обработки для каждого документа
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")
        
        results = []
        
        # Поддерживаемые форматы
        supported_formats = [".docx", ".txt", ".pdf"]
        
        for doc_file in folder.iterdir():
            if doc_file.is_file() and doc_file.suffix.lower() in supported_formats:
                try:
                    result = self.process_document(str(doc_file))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка при обработке {doc_file}: {e}", exc_info=True)
        
        logger.info(f"Обработано документов: {len(results)} из {len(list(folder.iterdir()))}")
        
        return results

