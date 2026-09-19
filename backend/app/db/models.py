"""
ORM model for persisted tick snapshots.

Columns exactly match the flat schema returned by GET /datasink/latest
(see app/api/datasink.py → _flatten_snapshot).
"""

from __future__ import annotations

from sqlalchemy import Integer, Float, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class TickRecord(Base):
    """One row per simulation tick, persisted to Postgres."""

    __tablename__ = "tick_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Time fields ──────────────────────────────────────────────────────
    timestamp: Mapped[str] = mapped_column(String, nullable=False, index=True)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    shift: Mapped[str] = mapped_column(String, nullable=False)

    # ── Flow sensors (J1–J16) ────────────────────────────────────────────
    flow_J1: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J3: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J4: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J5: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J6: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J7: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J8: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J9: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J10: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J11: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J12: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J13: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J14: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J15: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_J16: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ── Pressure sensors (J1–J16) ────────────────────────────────────────
    pressure_J1: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J3: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J4: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J5: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J6: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J7: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J8: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J9: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J10: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J11: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J12: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J13: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J14: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J15: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pressure_J16: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ── Machine production rates (M1–M8) ─────────────────────────────────
    production_M1: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    production_M2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    production_M3: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    production_M4: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    production_M5: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    production_M6: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    production_M7: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    production_M8: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ── Machine states (M1–M8) ───────────────────────────────────────────
    machine_status_M1: Mapped[str] = mapped_column(String, nullable=False, default="OFF")
    machine_status_M2: Mapped[str] = mapped_column(String, nullable=False, default="OFF")
    machine_status_M3: Mapped[str] = mapped_column(String, nullable=False, default="OFF")
    machine_status_M4: Mapped[str] = mapped_column(String, nullable=False, default="OFF")
    machine_status_M5: Mapped[str] = mapped_column(String, nullable=False, default="OFF")
    machine_status_M6: Mapped[str] = mapped_column(String, nullable=False, default="OFF")
    machine_status_M7: Mapped[str] = mapped_column(String, nullable=False, default="OFF")
    machine_status_M8: Mapped[str] = mapped_column(String, nullable=False, default="OFF")

    # ── Tap states (T1–T3) ───────────────────────────────────────────────
    tap_status_T1: Mapped[str] = mapped_column(String, nullable=False, default="CLOSED")
    tap_status_T2: Mapped[str] = mapped_column(String, nullable=False, default="CLOSED")
    tap_status_T3: Mapped[str] = mapped_column(String, nullable=False, default="CLOSED")

    # ── Leak labels (ground truth from injection) ────────────────────────
    leak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    leak_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    leak_zone: Mapped[str | None] = mapped_column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<TickRecord id={self.id} ts={self.timestamp}>"

    def to_flat_dict(self) -> dict:
        """Return the same flat dict shape as _flatten_snapshot."""
        d = {}
        for col in self.__table__.columns:
            if col.name == "id":
                continue
            d[col.name] = getattr(self, col.name)
        return d
