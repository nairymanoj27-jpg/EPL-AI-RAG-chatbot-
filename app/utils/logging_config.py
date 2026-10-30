"""Logging setup with request correlation and no secret/document logging."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, without leaking third-party debug output."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s [request_id=%(request_id)s] %(message)s"
            )
        )
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)
    root.setLevel(numeric_level)

    # These libraries can be extremely chatty at DEBUG. Keep them at WARNING
    # unless the application itself is being debugged.
    for noisy in ("httpx", "httpcore", "google_genai", "pinecone"):
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))
