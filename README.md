# Label Wise Server

Python backend for Label Wise data collection, installation registration, training-payload ingestion, and future model training/export workflows.

## Planned responsibilities
- Register app installations and issue server tokens
- Accept training payloads from the mobile app
- Enforce token-based request checks
- Store raw payloads and indexed metadata
- Provide admin APIs for inspection and exports
- Support future dataset preparation and training jobs

## Suggested stack
- FastAPI
- SQLAlchemy
- PostgreSQL
- Alembic
- Pydantic

## Initial structure
- `app/api/routes`: API route modules
- `app/core`: config and security helpers
- `app/db`: DB session/base setup
- `app/models`: ORM models
- `app/schemas`: request/response schemas
- `app/services`: business logic
- `scripts`: future data/export/training utilities
- `tests`: API and service tests
