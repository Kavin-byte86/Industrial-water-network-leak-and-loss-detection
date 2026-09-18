import { api } from '../../api/client';

const NAV_ITEMS = [
  { id: 'pipeline', label: 'Pipeline Skeleton', icon: PipelineIcon },
  { id: 'facility', label: 'Facility Map', icon: MapIcon },
  { id: 'liveflow', label: 'Live Flow Data', icon: DataIcon },
  { id: 'history', label: 'Historical Data', icon: HistoryIcon },
];

export function Sidebar({ activeView, onNavigate, state, connected }) {
  const timestamp = state?.timestamp || '—';
  const shift = state?.shift || '—';

  const handleStep = () => api.step();
  const handlePause = () => api.pause();
  const handleResume = () => api.resume();
  const handleReset = () => api.reset();

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
        <div className="sim-controls">
          <button className="sim-btn" onClick={handleStep}>Step</button>
          <button className="sim-btn" onClick={handlePause}>Pause</button>
          <button className="sim-btn" onClick={handleResume}>Resume</button>
          <button className="sim-btn" onClick={handleReset}>Reset</button>
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

function HistoryIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6" />
      <polyline points="8,4 8,8 11,10" />
    </svg>
  );
}
