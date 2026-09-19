"""
Database engine and session factory — reads DATABASE_URL from core/config.py.

Uses synchronous SQLAlchemy (psycopg2) for simplicity.  Writes from the tick
loop are dispatched to a background thread so they never block the asyncio
event loop.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase

from app.core.config import DATABASE_URL

logger = logging.getLogger(__name__)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set.  Add it as an environment variable "
        "(e.g. on Render's dashboard) or in backend/app/core/config.py.  "
        "The backend cannot start without a database connection."
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # reconnect on Neon idle-timeout
    pool_size=5,
    max_overflow=10,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def get_session() -> Session:
    """Return a new session (caller must close it)."""
    return SessionLocal()


def create_tables() -> None:
    """CREATE TABLE IF NOT EXISTS for every registered model."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured (create_all).")
