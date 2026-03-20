from fastapi import APIRouter

from app.api.routes.installations import router as installations_router
from app.api.routes.records import router as records_router

router = APIRouter()
router.include_router(installations_router)
router.include_router(records_router)
