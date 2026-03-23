from fastapi import APIRouter

from app.api.routes.distillation_jobs import router as distillation_jobs_router
from app.api.routes.installations import router as installations_router
from app.api.routes.model_versions import router as model_versions_router
from app.api.routes.records import router as records_router

router = APIRouter()
router.include_router(distillation_jobs_router)
router.include_router(installations_router)
router.include_router(model_versions_router)
router.include_router(records_router)
