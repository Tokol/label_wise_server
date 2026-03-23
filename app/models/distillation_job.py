from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DistillationJob(Base):
    __tablename__ = "distillation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    base_model: Mapped[str] = mapped_column(String(120), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_mode: Mapped[str] = mapped_column(String(64), nullable=False, default="single_batch")
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="queued", index=True)
    progress_stage: Mapped[str] = mapped_column(String(120), nullable=False, default="Queued")
    train_record_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    validation_record_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(400), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
