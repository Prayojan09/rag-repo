"""Small centralized logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    log_path = Path("data/logs/backend.log")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = str(log_path.resolve())
        if not any(
            isinstance(handler, RotatingFileHandler) and handler.baseFilename == resolved_path
            for handler in root_logger.handlers
        ):
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=2 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(file_handler)
    except OSError:
        root_logger.warning("File logging is unavailable; continuing with console logs")
