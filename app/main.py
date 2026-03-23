from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.router import router as api_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.models import analysis_record, distillation_job, installation, model_version  # noqa: F401


def _ensure_analysis_record_lifecycle_columns() -> None:
    inspector = inspect(engine)
    if "analysis_records" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("analysis_records")}
    statements = []
    if "distillation_status" not in existing_columns:
        statements.append("ALTER TABLE analysis_records ADD COLUMN distillation_status VARCHAR(64)")
    if "distillation_batch_id" not in existing_columns:
        statements.append("ALTER TABLE analysis_records ADD COLUMN distillation_batch_id VARCHAR(120)")
    if "reviewed_at" not in existing_columns:
        statements.append("ALTER TABLE analysis_records ADD COLUMN reviewed_at TIMESTAMP")
    if "exported_at" not in existing_columns:
        statements.append("ALTER TABLE analysis_records ADD COLUMN exported_at TIMESTAMP")
    if "used_in_training_at" not in existing_columns:
        statements.append("ALTER TABLE analysis_records ADD COLUMN used_in_training_at TIMESTAMP")
    if "excluded_reason" not in existing_columns:
        statements.append("ALTER TABLE analysis_records ADD COLUMN excluded_reason VARCHAR(200)")

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

        connection.execute(
            text(
                """
                UPDATE analysis_records
                SET distillation_status = CASE
                    WHEN distillation_status IS NOT NULL THEN distillation_status
                    WHEN usable_for_training THEN 'pending_review'
                    ELSE 'excluded'
                END
                WHERE distillation_status IS NULL
                """
            )
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    _ensure_analysis_record_lifecycle_columns()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://label-wsie-dashboard.onrender.com",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
