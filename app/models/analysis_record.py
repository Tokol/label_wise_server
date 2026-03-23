from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    installation_id: Mapped[str] = mapped_column(
        String(120), ForeignKey("installations.installation_id"), nullable=False, index=True
    )
    schema_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    route_type: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    platform: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    overall_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    usable_for_training: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    distillation_status: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    distillation_batch_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exported_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    used_in_training_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    excluded_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
