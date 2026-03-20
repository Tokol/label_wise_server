from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.installation import Installation
from app.schemas.installation import (
    InstallationRegisterRequest,
    InstallationRegisterResponse,
    InstallationSummary,
)
from app.services.security import generate_client_token, hash_token

router = APIRouter(prefix="/installations", tags=["installations"])


@router.post("/register", response_model=InstallationRegisterResponse, status_code=status.HTTP_201_CREATED)
def register_installation(payload: InstallationRegisterRequest, db: Session = Depends(get_db)):
    existing = db.get(Installation, payload.installation_id)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="installation_id is already registered",
        )

    client_token = generate_client_token()
    installation = Installation(
        installation_id=payload.installation_id,
        token_hash=hash_token(client_token),
        platform=payload.platform,
        app_version=payload.app_version,
    )
    db.add(installation)
    db.commit()
    db.refresh(installation)
    return InstallationRegisterResponse(
        installation_id=installation.installation_id,
        client_token=client_token,
        created_at=installation.created_at,
    )


@router.get("", response_model=list[InstallationSummary])
def list_installations(db: Session = Depends(get_db)):
    rows = db.scalars(select(Installation).order_by(Installation.created_at.desc())).all()
    return [
        InstallationSummary(
            installation_id=row.installation_id,
            platform=row.platform,
            app_version=row.app_version,
            created_at=row.created_at,
            last_seen_at=row.last_seen_at,
        )
        for row in rows
    ]
