import { useState, useEffect, useCallback } from 'react';
import { api } from '../../api/client';

const NAV_ITEMS = [
  { id: 'pipeline', label: 'Pipeline Skeleton', icon: PipelineIcon },
  { id: 'facility', label: 'Facility Map', icon: MapIcon },
  { id: 'liveflow', label: 'Live Flow Data', icon: DataIcon },
  { id: 'analysis', label: 'Production Analysis', icon: AnalysisIcon },
  { id: 'history', label: 'Historical Data', icon: HistoryIcon },
];

export function Sidebar({ activeView, onNavigate, state, connected, onError }) {
  const timestamp = state?.timestamp || '—';
  const shift = state?.shift || '—';

  // Mirrors the engine's real pause state instead of assuming it, so the label
  // is still correct after a page reload.
  const [paused, setPaused] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!connected) return undefined;
    api.getSimStatus()
      .then(s => { if (!cancelled) setPaused(Boolean(s.paused)); })
      .catch(() => { /* status is advisory; the banner covers real outages */ });
    return () => { cancelled = true; };
  }, [connected]);

  // Every control used to fire a floating promise: a failure produced an
  // unhandled rejection in the console and no visible feedback at all.
  const run = useCallback(async (action, after) => {
    setBusy(true);
    try {
      const result = await action();
      if (after) after(result);
      onError?.(null);
    } catch (err) {
      onError?.(err.message);
    } finally {
      setBusy(false);
    }
  }, [onError]);

  const handleStep = () => run(api.step);
  const handlePause = () => run(api.pause, () => setPaused(true));
  const handleResume = () => run(api.resume, () => setPaused(false));
  const handleReset = () => run(api.reset);

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>ABC Industries</h1>
        <span>Water Network Monitor (Textile Manufacturing Sector)</span>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(item => (
          <div
            key={item.id}
            className={`nav-item ${activeView === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            <item.icon />
            <span>{item.label}</span>
          </div>
        ))}
      </nav>

      <div className="status-strip">
        <div className="status-row">
          <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
          <span className="status-label">Backend</span>
          <span className="status-value">{connected ? 'Connected' : 'Offline'}</span>
        </div>
        <div className="status-row">
          <span className="status-label">Shift</span>
          <span className="status-value">{shift}</span>
        </div>
        <div className="status-row">
          <span className="status-label">Time</span>
          <span className="status-value">{timestamp.replace('T', ' ')}</span>
        </div>
        <div className="status-row">
          <span className="status-label">Sim</span>
          <span className="status-value">{paused ? 'Paused' : 'Running'}</span>
        </div>
        <div className="sim-controls">
          <button className="sim-btn" onClick={handleStep}
            disabled={busy || !connected}
            title="Advance exactly one tick (5 simulated minutes)">Step</button>
          {paused ? (
            <button className="sim-btn" onClick={handleResume}
              disabled={busy || !connected}
              title="Resume the auto-tick loop">Resume</button>
          ) : (
            <button className="sim-btn" onClick={handlePause}
              disabled={busy || !connected}
              title="Pause the auto-tick loop">Pause</button>
          )}
          <button className="sim-btn" onClick={handleReset}
            disabled={busy || !connected}
            title="All machines OFF, taps CLOSED, clock reset">Reset</button>
        </div>
      </div>
    </aside>
  );
}

/* ── Inline SVG Icons ── */
function PipelineIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="3" r="2" /><line x1="8" y1="5" x2="4" y2="10" />
      <line x1="8" y1="5" x2="12" y2="10" /><circle cx="4" cy="12" r="2" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}

function MapIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="1" y="1" width="14" height="14" rx="2" />
      <rect x="3" y="4" width="4" height="3" rx="0.5" />
      <rect x="9" y="4" width="4" height="3" rx="0.5" />
      <rect x="3" y="9" width="4" height="3" rx="0.5" />
      <rect x="9" y="9" width="4" height="3" rx="0.5" />
    </svg>
  );
}

function DataIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <polyline points="1,12 4,6 7,9 10,3 13,7 15,4" />
      <line x1="1" y1="14" x2="15" y2="14" />
    </svg>
  );
}

function AnalysisIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <line x1="2" y1="13" x2="14" y2="13" />
      <rect x="3" y="8" width="3" height="5" />
      <rect x="7.5" y="4" width="3" height="9" />
      <rect x="12" y="10" width="2.5" height="3" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6" />
      <polyline points="8,4 8,8 11,10" />
    </svg>
  );
}
