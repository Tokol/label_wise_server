from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Label Wise Server"
    app_version: str = "0.1.0"
    database_url: str = f"sqlite:///{Path(__file__).resolve().parents[2] / 'label_wise.db'}"
    api_prefix: str = "/api"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
