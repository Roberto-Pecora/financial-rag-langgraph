"""Structured logging: JSON in production, console in dev. Configured once."""

from __future__ import annotations

import logging

import structlog

from frag.utils.settings import settings

_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(format="%(message)s", level=settings.log_level.upper())
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure()
    return structlog.get_logger(name)
