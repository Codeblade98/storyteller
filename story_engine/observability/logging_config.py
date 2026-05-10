import logging
import logging.config
import os
from pathlib import Path


_CONFIGURED = False


def configure_logging(log_dir: str | Path | None = None, level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_log_dir = Path(log_dir or os.getenv("STORY_ENGINE_LOG_DIR", "logs"))
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_level = (level or os.getenv("STORY_ENGINE_LOG_LEVEL", "INFO")).upper()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                },
                "app_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                    "filename": str(resolved_log_dir / "app.log"),
                    "maxBytes": 1_000_000,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "api_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                    "filename": str(resolved_log_dir / "api.log"),
                    "maxBytes": 1_000_000,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "engine_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                    "filename": str(resolved_log_dir / "engine.log"),
                    "maxBytes": 2_000_000,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "llm_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                    "filename": str(resolved_log_dir / "llm.log"),
                    "maxBytes": 2_000_000,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "verification_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                    "filename": str(resolved_log_dir / "verification.log"),
                    "maxBytes": 2_000_000,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "WARNING",
                    "formatter": "standard",
                    "filename": str(resolved_log_dir / "error.log"),
                    "maxBytes": 2_000_000,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "story_engine": {
                    "handlers": ["console", "app_file", "error_file"],
                    "level": resolved_level,
                    "propagate": False,
                },
                "story_engine.api": {
                    "handlers": ["console", "api_file", "error_file"],
                    "level": resolved_level,
                    "propagate": False,
                },
                "story_engine.engine": {
                    "handlers": ["console", "engine_file", "error_file"],
                    "level": resolved_level,
                    "propagate": False,
                },
                "story_engine.llm": {
                    "handlers": ["console", "llm_file", "error_file"],
                    "level": resolved_level,
                    "propagate": False,
                },
                "story_engine.verification": {
                    "handlers": ["console", "verification_file", "error_file"],
                    "level": resolved_level,
                    "propagate": False,
                },
            },
            "root": {
                "handlers": ["console", "app_file", "error_file"],
                "level": "WARNING",
            },
        }
    )
    logging.getLogger("story_engine").info(
        "logging_configured log_dir=%s level=%s files=app,api,engine,llm,verification,error",
        resolved_log_dir,
        resolved_level,
    )
    _CONFIGURED = True
