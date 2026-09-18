"""
FastAPI application entry point.

Wires up all routers, and starts the background auto-tick loop on startup.
The shared store and engine instances live in app/dependencies.py.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.control import router as control_router
from app.api.datasink import router as datasink_router
from app.api.network import router as network_router
from app.api.predict import router as predict_router
from app.api.simulation import router as simulation_router
from app.api.state import router as state_router
from app.api.testbench import router as testbench_router
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
    title="ABC Industries Water Network Simulation",
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
app.include_router(predict_router)
app.include_router(testbench_router)


@app.get("/health", tags=["health"])
def health_check():
    """Health-check endpoint.

    Lives at /health rather than / because / serves the built frontend when one
    is present (see below).
    """
    return {"status": "ok", "service": "water-network-simulation"}


# ── Serve the built frontend (single-origin deployment) ───────────────────
# When frontend/dist exists, mount it at / so one process serves both the API
# and the UI on one port. No CORS, no proxy, no second server.
#
# Build it with:  cd frontend && npm run build
#
# This mount is registered LAST on purpose: Starlette matches routes in
# registration order, so every API route above still wins over the catch-all.
# html=True makes unmatched paths fall back to index.html, which is what a
# single-page app needs.
_FRONTEND_DIST = Path(
    os.getenv(
        "FRONTEND_DIST",
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"),
    )
).resolve()

if (_FRONTEND_DIST / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIST), html=True),
        name="frontend",
    )
else:
    @app.get("/", tags=["health"])
    def frontend_not_built():
        """Placeholder while no production frontend build exists."""
        return {
            "status": "ok",
            "service": "water-network-simulation",
            "detail": (
                "No frontend build found. Run 'cd frontend && npm run build' to "
                "serve the UI from this origin, or use the Vite dev server on "
                "port 5173."
            ),
            "expected_path": str(_FRONTEND_DIST),
        }
