"""
FastAPI приложение эмулятора КФО.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from desktop.cfx_emulator.routes import file, irv, user
from desktop.cfx_emulator import storage

# Создаем приложение
app = FastAPI(
    title="CFX Emulator",
    description="Эмулятор КФО для отладки Scrivener в автономном режиме",
    version="0.1.0",
)

# Настраиваем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роуты
app.include_router(user.router, prefix="/siu-star/services/api")
app.include_router(irv.router, prefix="/siu-star/services/api")
app.include_router(file.router, prefix="/siu-star/services/api")


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске."""
    storage.ensure_storage_dir()


@app.get("/")
async def root():
    """Корневой endpoint."""
    return {
        "message": "CFX Emulator",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Проверка здоровья сервиса."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

