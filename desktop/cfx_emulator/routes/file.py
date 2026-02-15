"""
Роуты для работы с файлами.
"""

from typing import Optional

from fastapi import APIRouter, Cookie, HTTPException, Query, Request
from fastapi.responses import Response

from desktop.cfx_emulator import storage

router = APIRouter()


@router.get("/file/{irvf_id}/read")
async def read_file(
    irvf_id: str,
    request: Request,
    JSESSIONID: str = Cookie(None),
):
    """
    Чтение содержимого файла.
    
    Возвращает текст для текстовых файлов или байты для бинарных.
    """
    content = storage.get_file_content(irvf_id)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Файл {irvf_id} не найден")
    
    # Определяем тип файла по расширению (если есть в пути)
    # Для простоты возвращаем байты, клиент сам определит тип
    return Response(content=content, media_type="application/octet-stream")


@router.post("/file/{irvf_id}/write")
async def write_file(
    irvf_id: str,
    request: Request,
    fileName: str = Query(...),
    crc: Optional[str] = Query(None),
    irvId: Optional[str] = Query(None),
    JSESSIONID: str = Cookie(None),
):
    """
    Запись содержимого файла.
    
    Args:
        irvf_id: UUID версии файла
        fileName: Имя файла
        crc: Контрольная сумма (опционально)
        irvId: UUID версии ИО (опционально, если не указан - создается новый ИО)
        request: Request с телом файла
    """
    content = await request.body()
    
    # Сохраняем файл
    irv_id = storage.save_file(irvf_id, fileName, content, irv_id=irvId)
    
    return {
        "success": True,
        "irvfId": irvf_id,
        "irvId": irv_id,
        "fileName": fileName,
    }

