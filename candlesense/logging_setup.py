"""Logs no console e em arquivo rotativo."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .config import Config, app_dir


def setup_logging(cfg: Config) -> Path | None:
    level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    log_path: Path | None = None
    try:
        log_dir = Path(cfg.logging.dir)
        if not log_dir.is_absolute():
            log_dir = app_dir() / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "candlesense.log"
        fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        logging.getLogger(__name__).warning("não foi possível criar o arquivo de log")
    return log_path
