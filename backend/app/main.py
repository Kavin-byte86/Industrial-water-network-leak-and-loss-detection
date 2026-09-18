"""
FastAPI application entry point.

Wires up all routers, and starts the background auto-tick loop on startup.
The shared store and engine instances live in app/dependencies.py.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.control import router as control_router
from app.api.datasink import router as datasink_router
from app.api.network import router as network_router
from app.api.simulation import router as simulation_router
from app.api.state import router as state_router
from app.dependencies import engine


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tick loop
    await engine.start_background_loop()
    yield
    # Shutdown
    await engine.stop()


# ── App factory ───────────────────────────────────────────────────────────
app = FastAPI(
    title="ABC Textiles Water Network Simulation",
    description=(
        "Simulates a 16-junction, 8-machine, 3-tap textile-mill water "
        "network for leak-detection development."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — open for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────
app.include_router(network_router)
app.include_router(state_router)
app.include_router(control_router)
app.include_router(simulation_router)
app.include_router(datasink_router)


@app.get("/", tags=["health"])
def health_check():
    """Root health-check endpoint."""
    return {"status": "ok", "service": "water-network-simulation"}
