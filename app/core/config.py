from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Label Wise Server"
    app_version: str = "0.1.0"
    database_url: str = f"sqlite:///{Path(__file__).resolve().parents[2] / 'label_wise.db'}"
    api_prefix: str = "/api"
    worker_poll_seconds: float = 5.0
    worker_base_url: str = "http://127.0.0.1:8000"
    worker_id: str = "label-wise-worker"
    worker_artifacts_dir: str = str(Path(__file__).resolve().parents[2] / "artifacts")
    trainer_python_bin: str = "python3"
    trainer_module: str = "app.services.training_runner"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url.startswith("postgres://"):
            return self.database_url.replace("postgres://", "postgresql+psycopg://", 1)
        if self.database_url.startswith("postgresql://") and "+psycopg" not in self.database_url:
            return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        return self.database_url


settings = Settings()
