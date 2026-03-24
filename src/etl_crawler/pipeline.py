"""Pipeline runner that orchestrates ETL steps."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from src.etl_crawler.blob_artifacts import ETLArtifactUploader
from src.etl_crawler.config import AppSettings, ETLSettings
from src.etl_crawler.steps import crawl, extract, post_process, index

logger = logging.getLogger(__name__)

CUSTOMER_CONFIGS_DIR = Path(__file__).parent / "customer_configs"

STEP_REGISTRY: dict[str, Any] = {
    "crawl": crawl.run,
    "extract": extract.run,
    "post_process": post_process.run,
    "index": index.run,
}


def _enforce_quiet_dependency_logging() -> None:
    """Clamp noisy third-party loggers to reduce terminal spam."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        handler.setLevel(logging.INFO)

    noisy_loggers = (
        "pdfminer",
        "pdfplumber",
        "httpcore",
        "httpx",
        "urllib3",
        "openai",
        "azure",
        "azure.core.pipeline.policies.http_logging_policy",
    )
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


@dataclass
class RunContext:
    """Shared context passed to every pipeline step."""

    customer_name: str
    data_dir: Path
    customer_config: dict
    app_settings: AppSettings


def load_customer_config(customer_name: str) -> dict:
    """Load a customer YAML config by name."""
    config_path = CUSTOMER_CONFIGS_DIR / f"{customer_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No customer config found at {config_path}. "
            f"Available configs: {[p.stem for p in CUSTOMER_CONFIGS_DIR.glob('*.yml')]}"
        )
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_data_dir(customer_name: str, data_dir: Path | None = None) -> Path:
    """Resolve or create a timestamped data directory for the run.

    If *data_dir* is given and already exists (e.g. resuming a previous run),
    it is returned as-is.  Otherwise a new timestamped directory is created.
    """
    if data_dir and data_dir.exists():
        return data_dir

    base = Path("data") / customer_name
    if data_dir is None:
        data_dir = base / str(int(time.time()))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def run_pipeline(
    customer_name: str,
    steps: list[str] | None = None,
    data_dir: Path | None = None,
) -> None:
    """Run the ETL pipeline for a customer.

    Args:
        customer_name: Name matching a YAML config in customer_configs/.
        steps: Optional subset of steps to run (default: all steps in order).
        data_dir: Optional explicit data directory (default: new timestamped dir).
    """
    _enforce_quiet_dependency_logging()
    data_dir = resolve_data_dir(customer_name, data_dir)
    customer_config = load_customer_config(customer_name)
    etl_settings = ETLSettings()
    app_settings = AppSettings()
    env_label = app_settings.ENV or "dev"
    artifact_uploader: ETLArtifactUploader | None = None

    if (
        etl_settings.AZURE_STORAGE_ACCOUNT_PRIMARY_CONNECTION_STRING
        and etl_settings.AZURE_CONTAINER_STORAGE_ETL_FILES_NAME
    ):
        run_prefix = f"{env_label}/{customer_name}/{data_dir.name}"
        artifact_uploader = ETLArtifactUploader(
            connection_string=etl_settings.AZURE_STORAGE_ACCOUNT_PRIMARY_CONNECTION_STRING,
            container_name=etl_settings.AZURE_CONTAINER_STORAGE_ETL_FILES_NAME,
            run_prefix=run_prefix,
        )
        logger.info(
            "Artifact uploads enabled: container=%s run_prefix=%s",
            etl_settings.AZURE_CONTAINER_STORAGE_ETL_FILES_NAME,
            run_prefix,
        )
    else:
        logger.warning(
            "Artifact uploads disabled. Missing AZURE_STORAGE_ACCOUNT_PRIMARY_CONNECTION_STRING "
            "or AZURE_CONTAINER_STORAGE_ETL_FILES_NAME."
        )

    run_context = RunContext(
        customer_name=customer_name,
        data_dir=data_dir,
        customer_config=customer_config,
        app_settings=app_settings,
    )

    requested_steps = steps or list(STEP_REGISTRY)

    for step_name in requested_steps:
        if step_name not in STEP_REGISTRY:
            raise ValueError(
                f"Unknown step '{step_name}'. Available: {list(STEP_REGISTRY)}"
            )

    logger.info("Pipeline starting for customer=%s data_dir=%s steps=%s",
                customer_name, data_dir, requested_steps)

    for step_name in requested_steps:
        _enforce_quiet_dependency_logging()
        logger.info("Running step: %s", step_name)
        STEP_REGISTRY[step_name](run_context)
        if artifact_uploader:
            artifact_uploader.sync_run_outputs(run_context.data_dir)
        logger.info("Completed step: %s", step_name)

    logger.info("Pipeline finished.")
