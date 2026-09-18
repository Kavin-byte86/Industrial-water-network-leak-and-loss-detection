"""
Shared application dependencies — singleton instances of the session store
and simulation engine, importable by both routers and main.py without
circular imports.
"""

from app.simulation.engine import SimulationEngine
from app.state.session_store import SessionStore

store = SessionStore()
engine = SimulationEngine(store)
