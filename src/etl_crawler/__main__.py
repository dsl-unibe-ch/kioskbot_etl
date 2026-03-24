"""CLI entrypoint: ``python -m etl_crawler run --customer quality [--steps crawl,extract]``."""

import argparse
import logging
from pathlib import Path

from src.etl_crawler.pipeline import STEP_REGISTRY, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)

# Keep noisy dependency loggers from flooding terminal output.
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("pdfplumber").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="etl_crawler",
        description="Run the ETL crawler pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute pipeline steps")
    run_parser.add_argument(
        "--customer",
        required=True,
        help="Customer name (must match a YAML config in customer_configs/).",
    )
    run_parser.add_argument(
        "--steps",
        default=None,
        help=f"Comma-separated list of steps to run (default: all). Available: {', '.join(STEP_REGISTRY)}",
    )
    run_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Explicit data directory (default: data/<customer>/<timestamp>).",
    )

    args = parser.parse_args()

    if args.command == "run":
        steps = args.steps.split(",") if args.steps else None
        run_pipeline(
            customer_name=args.customer,
            steps=steps,
            data_dir=args.data_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
