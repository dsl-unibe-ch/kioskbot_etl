"""
Configuration Settings for the ETL Crawler
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

_env = os.getenv("ENV", "")

_etl_env_file = f".env.etl"
_app_env_file = f".env.{_env}.app" if _env else ".env.dev.app"

class ETLSettings(BaseSettings):
    """
    Settings for the ETL crawler which are loaded first and required to download app secrets from the container
    """
    AZURE_STORAGE_ACCOUNT_PRIMARY_CONNECTION_STRING: str | None = Field(None, description="The primary connection string for the Azure Storage Account")
    AZURE_CONTAINER_STORAGE_NAME: str | None = None
    AZURE_CONTAINER_STORAGE_SECRETS_NAME: str | None = None
    AZURE_CONTAINER_STORAGE_ETL_FILES_NAME: str | None = None
    model_config = SettingsConfigDict(
        env_file=_etl_env_file,
        env_file_encoding="utf-8",
        extra="ignore",
    )

class AppSettings(BaseSettings):
    """
    Settings for the application which are loaded after the ETL settings are loaded and used to run the crawler
    """
    ENV: str | None = Field(None, description="The environment to run the crawler in")
    AZURE_SEARCH_SERVICE_PRIMARY_ADMIN_KEY: str
    AZURE_OPENAI_PRIMARY_KEY: str
    AZURE_SEARCH_ENDPOINT: str
    AZURE_DEFAULT_AI_SEARCH_INDEX_NAME: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_SEARCH_EMBEDDING_DEPLOYMENT: str
    AZURE_OPENAI_SEARCH_EMBEDDING_API_VERSION: str
    AZURE_OPENAI_CHAT_DEPLOYMENT: str
    AZURE_OPENAI_CHAT_API_VERSION: str
    AZURE_OPENAI_VECTORIZER_ENDPOINT: str
    

    model_config = SettingsConfigDict(env_file=_app_env_file, env_file_encoding="utf-8", extra="ignore")