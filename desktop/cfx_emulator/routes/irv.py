"""
Роуты для работы с информационными объектами (ИО).
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Cookie, HTTPException

from desktop.cfx_emulator import storage

router = APIRouter()


@router.get("/irv/{irv_id}")
async def get_irv_info(irv_id: str, JSESSIONID: str = Cookie(None)):
    """
    Получение краткой информации об ИО.
    """
    metadata = storage.load_metadata(irv_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"ИО {irv_id} не найден")
    
    return {
        "irv_id": irv_id,
        "name": metadata.get("name", ""),
        "description": metadata.get("description", ""),
        "ir": metadata.get("ir", {}),
    }


@router.post("/irv/{irv_id}")
async def get_irv_full(
    irv_id: str,
    body: Dict[str, Any] = Body(...),
    JSESSIONID: str = Cookie(None),
):
    """
    Получение полной информации об ИО с параметрами.
    """
    params = {
        "with_meta": body.get("withMeta", True),
        "with_base_metas": body.get("withBaseMetas", True),
        "with_files": body.get("withFiles", True),
        "with_semantic": body.get("withSemantic", True),
        "plane_values": body.get("planeValues", False),
        "with_dict_childs": body.get("withDictChilds", False),
        "with_dict_childs_as_object": body.get("withDictChildsAsObject", False),
        "depth": body.get("depth", 0),
        "dict_sort_order": body.get("dictSortOrder", "name"),
    }
    
    result = storage.get_irv(irv_id, **params)
    if not result:
        raise HTTPException(status_code=404, detail=f"ИО {irv_id} не найден")
    
    return result


@router.get("/irv/{irv_id}/files")
async def get_irv_files_list(irv_id: str, JSESSIONID: str = Cookie(None)):
    """
    Получение списка файлов ИО.
    """
    # Проверяем существование ИО
    metadata = storage.load_metadata(irv_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"ИО {irv_id} не найден")
    
    files = storage.get_irv_files(irv_id)
    return files


@router.post("/folder/{folder_id}/irvs")
async def create_irv(
    folder_id: str,
    irv_data: Dict[str, Any] = Body(...),
    JSESSIONID: str = Cookie(None),
):
    """
    Создание нового ИО в папке.
    """
    irv_data["parentId"] = folder_id
    result = storage.create_irv(irv_data)
    return result

