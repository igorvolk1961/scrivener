"""
API клиент для работы с Scrivener API.
"""

import json
from typing import Any, Dict, Optional

import httpx
from loguru import logger


class ScrivenerClient:
    """Клиент для работы с API Scrivener."""
    
    def __init__(self, api_url: str, cfx_emulator_url: str = "http://localhost:8001"):
        """
        Инициализация клиента.
        
        Args:
            api_url: URL API Scrivener (например, http://localhost:8000)
            cfx_emulator_url: URL эмулятора КФО (по умолчанию http://localhost:8001)
        """
        self.api_url = api_url.rstrip("/")
        self.cfx_emulator_url = cfx_emulator_url.rstrip("/")
        # Таймаут ожидания ответа от API (generate, RAG и т.д.) — 5 минут
        self.timeout = 10 * 60.0  # 600 секунд
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет HTTP запрос к API.
        
        Args:
            method: HTTP метод (GET, POST, DELETE)
            endpoint: Endpoint API (например, /v1/generate)
            data: Тело запроса (для POST)
            params: Query параметры
        
        Returns:
            Словарь с результатом: {"success": bool, "data": ..., "error": ...}
        """
        url = f"{self.api_url}{endpoint}"
        
        # Заголовки для работы с эмулятором КФО
        headers = {
            "Content-Type": "application/json",
            "Referer": self.cfx_emulator_url,
        }
        cookies = {"JSESSIONID": "debug"}
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                if method == "GET":
                    response = client.get(url, headers=headers, cookies=cookies, params=params)
                elif method == "POST":
                    response = client.post(url, headers=headers, cookies=cookies, json=data, params=params)
                elif method == "DELETE":
                    response = client.delete(url, headers=headers, cookies=cookies, params=params)
                else:
                    return {
                        "success": False,
                        "error": f"Неподдерживаемый метод: {method}"
                    }
                
                response.raise_for_status()
                result = response.json()
                
                # Проверяем наличие ошибки в ответе
                if "error" in result:
                    return {
                        "success": False,
                        "error": result.get("error"),
                        "detail": result.get("detail"),
                        "code": result.get("code"),
                        "data": result,
                    }
                
                return {
                    "success": True,
                    "data": result,
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка {e.response.status_code}: {e.response.text}")
            try:
                error_data = e.response.json()
                return {
                    "success": False,
                    "error": error_data.get("error", "HTTP ошибка"),
                    "detail": error_data.get("detail", e.response.text),
                    "code": error_data.get("code", "http_error"),
                }
            except:
                return {
                    "success": False,
                    "error": "HTTP ошибка",
                    "detail": f"{e.response.status_code}: {e.response.text}",
                    "code": "http_error",
                }
        except httpx.RequestError as e:
            logger.error(f"Ошибка запроса: {e}")
            error_str = str(e)
            # Улучшаем сообщение об ошибке
            if "Connection refused" in error_str:
                error_msg = "Соединение отклонено"
            elif "timeout" in error_str.lower():
                error_msg = "Таймаут соединения"
            elif "Name resolution" in error_str or "getaddrinfo" in error_str:
                error_msg = "Не удалось разрешить имя хоста"
            else:
                error_msg = "Ошибка соединения"
            
            return {
                "success": False,
                "error": error_msg,
                "detail": error_str,
                "code": "connection_error",
            }
        except Exception as e:
            logger.exception(f"Неожиданная ошибка: {e}")
            return {
                "success": False,
                "error": "Неожиданная ошибка",
                "detail": str(e),
                "code": "unexpected_error",
            }
    
    def generate_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерация ответа от LLM.
        
        Args:
            request_data: Данные запроса (AssistantRequest)
        
        Returns:
            Результат генерации
        """
        return self._make_request("POST", "/v1/generate", data=request_data)
    
    def manage_rag_files(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Управление файлами в RAG.
        
        Args:
            request_data: Данные запроса (RAGRequest)
        
        Returns:
            Результат операции
        """
        return self._make_request("POST", "/v1/rag/manage", data=request_data)
    
    def check_qdrant_health(self, vdb_url: str) -> Dict[str, Any]:
        """
        Проверка доступности Qdrant.
        
        Args:
            vdb_url: URL векторной базы данных
        
        Returns:
            Статус доступности
        """
        request_data = {"vdb_url": vdb_url}
        return self._make_request("POST", "/v1/rag/health", data=request_data)
    
    def get_collections(self, vdb_url: str) -> Dict[str, Any]:
        """
        Получение списка коллекций.
        
        Args:
            vdb_url: URL векторной базы данных
        
        Returns:
            Список коллекций
        """
        request_data = {"vdb_url": vdb_url}
        return self._make_request("POST", "/v1/rag/collections", data=request_data)
    
    def delete_collection(self, collection_name: str, vdb_url: str) -> Dict[str, Any]:
        """
        Удаление коллекции.
        
        Args:
            collection_name: Имя коллекции
            vdb_url: URL векторной базы данных
        
        Returns:
            Результат удаления
        """
        params = {"vdb_url": vdb_url}
        return self._make_request("DELETE", f"/v1/rag/collections/{collection_name}", params=params)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Получение информации о кэше.
        
        Returns:
            Информация о кэше
        """
        return self._make_request("GET", "/v1/cache/info")
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Очистка кэша.
        
        Returns:
            Результат операции
        """
        return self._make_request("DELETE", "/v1/cache/clear")

