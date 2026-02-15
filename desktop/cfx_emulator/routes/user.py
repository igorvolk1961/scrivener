"""
Роуты для работы с пользователями.
"""

from fastapi import APIRouter, Cookie

from desktop.cfx_emulator import storage

router = APIRouter()


@router.get("/user/current")
async def get_current_user(JSESSIONID: str = Cookie(None)):
    """
    Получение информации о текущем пользователе.
    
    Возвращает информацию из хранилища эмулятора.
    """
    return storage.get_user_info()

