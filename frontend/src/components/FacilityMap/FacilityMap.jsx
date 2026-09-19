/**
 * View 2 — Facility Map (2D top-down factory floor layout)
 *
 * Spatial layout resembling an actual textile mill floor.
 * Machines placed in production zones, taps at utility areas.
 * Click a machine/tap to open inline control panel.
 */

import { useState, useRef, useCallback } from 'react';
import { api } from '../../api/client';
import { isFlowing, pipeClass } from '../../lib/flowState';

// Spatial layout — machines grouped by process zone
const LAYOUT = {
  // Zone A: Pre-treatment
  M1: { x: 100, y: 100, label: 'Scouring', zone: 'Pre-treatment' },
  M2: { x: 260, y: 100, label: 'Bleaching', zone: 'Pre-treatment' },
  // Zone B: Dyeing & Washing
  M3: { x: 100, y: 230, label: 'Dye Unit 1', zone: 'Dyeing' },
  M4: { x: 260, y: 230, label: 'Dye Unit 2', zone: 'Dyeing' },
  M5: { x: 420, y: 230, label: 'Washing', zone: 'Dyeing' },
  // Zone C: Finishing
  M6: { x: 100, y: 360, label: 'Finishing', zone: 'Finishing' },
  M7: { x: 260, y: 360, label: 'Calendering', zone: 'Finishing' },
  // Zone D: Utility
  M8: { x: 420, y: 360, label: 'Boiler Feed', zone: 'Utility' },
  // Taps
  T1: { x: 580, y: 130, label: 'Tap 1', zone: 'Utility' },
  T2: { x: 580, y: 260, label: 'Tap 2', zone: 'Utility' },
  T3: { x: 580, y: 390, label: 'Tap 3', zone: 'Utility' },
};

// Junction-to-endpoint mapping for pipe drawing
const JUNCTION_MAP = {
  M1: 'J5', M2: 'J6', M3: 'J8', M4: 'J9', M5: 'J10',
  M6: 'J11', M7: 'J12', M8: 'J13', T1: 'J14', T2: 'J15', T3: 'J16',
};

// Main supply pipe path
const SUPPLY_POINTS = [
  { x: 350, y: 30 },  // Main inlet (J1)
  { x: 350, y: 60 },
];

const MACHINE_STATES = ['OFF', 'STARTING', 'RUNNING', 'STOPPING', 'MAINTENANCE'];

const PANEL = { width: 200, height: 190, margin: 8 };

export function FacilityMap({ state, onError, leakNodes = [] }) {
  const [selected, setSelected] = useState(null);
  const [controlState, setControlState] = useState({ pct: 100, state: 'RUNNING' });
  const [panelPos, setPanelPos] = useState({ left: 0, top: 0 });
  const [busy, setBusy] = useState(false);
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const leaking = new Set(leakNodes);
  const flows = state?.flows || {};
  const machines = state?.machines || {};
  const taps = state?.taps || {};

  /**
   * Convert a point in SVG viewBox units to pixel coordinates inside the
   * container.
   *
   * The panel is absolutely positioned HTML, but LAYOUT holds viewBox units
   * (0-700 x 0-460). The SVG is scaled to fill the container and letterboxed by
   * preserveAspectRatio, so viewBox units are NOT pixels — using them directly
   * placed the panel away from the node that was clicked. getScreenCTM gives
   * the live transform, which stays correct at any window size.
   */
  const placePanel = useCallback((pos) => {
    const svg = svgRef.current;
    const container = containerRef.current;
    if (!svg || !container || !svg.getScreenCTM) return;

    const ctm = svg.getScreenCTM();
    if (!ctm) return;

    const pt = svg.createSVGPoint();
    pt.x = pos.x;
    pt.y = pos.y;
    const screen = pt.matrixTransform(ctm);
    const rect = container.getBoundingClientRect();

    // Offset slightly down-right of the node, then clamp so the panel can never
    // hang outside the visible map.
    const maxLeft = rect.width - PANEL.width - PANEL.margin;
    const maxTop = rect.height - PANEL.height - PANEL.margin;
    const left = screen.x - rect.left + 20;
    const top = screen.y - rect.top + 20;

    setPanelPos({
      left: Math.max(PANEL.margin, Math.min(left, Math.max(PANEL.margin, maxLeft))),
      top: Math.max(PANEL.margin, Math.min(top, Math.max(PANEL.margin, maxTop))),
    });
  }, []);

  // Control writes used to be un-caught: a rejected request left the panel open
  // with no explanation.
  const runControl = useCallback(async (action) => {
    setBusy(true);
    try {
      await action();
      onError?.(null);
      setSelected(null);
    } catch (err) {
      onError?.(err.message);
    } finally {
      setBusy(false);
    }
  }, [onError]);

  const handleApplyMachine = (mid) =>
    runControl(() => api.controlMachine(mid, controlState.pct, controlState.state));

  const handleApplyTap = (tid, newState) =>
    runControl(() => api.controlTap(tid, newState));

  return (
    <div>
      <div className="view-header">
        <h2 className="view-title">Facility Map</h2>
      </div>
      <div className="facility-map" ref={containerRef} style={{ position: 'relative' }}>
        <svg ref={svgRef} viewBox="0 0 700 460" preserveAspectRatio="xMidYMid meet"
          style={{ width: '100%', height: 'calc(100vh - 100px)', background: 'var(--bg-panel)', border: '1px solid var(--border-color)', borderRadius: '10px' }}>

          {/* Background zones */}
          <rect x="60" y="60" width="370" height="90" rx="6" fill="rgba(34,211,238,0.05)" stroke="#27405e" strokeDasharray="4" />
          <text x="245" y="78" textAnchor="middle" fontSize="8" fill="var(--text-muted)" fontWeight="700">PRE-TREATMENT</text>

          <rect x="60" y="190" width="430" height="90" rx="6" fill="rgba(34,211,238,0.05)" stroke="#27405e" strokeDasharray="4" />
          <text x="275" y="208" textAnchor="middle" fontSize="8" fill="var(--text-muted)" fontWeight="700">DYEING & WASHING</text>

          <rect x="60" y="320" width="430" height="90" rx="6" fill="rgba(34,211,238,0.05)" stroke="#27405e" strokeDasharray="4" />
          <text x="275" y="338" textAnchor="middle" fontSize="8" fill="var(--text-muted)" fontWeight="700">FINISHING & UTILITY</text>

          <rect x="540" y="90" width="120" height="340" rx="6" fill="rgba(34,197,94,0.05)" stroke="#2c4a3d" strokeDasharray="4" />
          <text x="600" y="108" textAnchor="middle" fontSize="8" fill="var(--text-muted)" fontWeight="700">TAPS</text>

          {/* ═══ PIPE NETWORK ═══ */}
          {/* Main supply line from inlet */}
          <text x="350" y="22" textAnchor="middle" fontSize="9" fontWeight="700" fill="var(--accent)">
            MAIN INLET (J1): {(flows['J1'] || 0).toFixed(0)} L/min
          </text>
          <line x1="350" y1="30" x2="350" y2="55"
            className={pipeClass(flows['J1'])}
            style={isFlowing(flows['J1']) ? { animationDuration: '0.8s' } : {}} />

          {/* Main horizontal trunk line */}
          <line x1="70" y1="55" x2="620" y2="55"
            className={isFlowing(flows['J1']) ? 'pipe-flow' : 'pipe-idle'}
            style={isFlowing(flows['J1']) ? { animationDuration: '0.6s', strokeWidth: 3 } : { strokeWidth: 2 }} />

          {/* Branch to taps zone */}
          <line x1="580" y1="55" x2="580" y2="410"
            className={isFlowing(flows['J7']) ? 'pipe-flow' : 'pipe-idle'}
            style={isFlowing(flows['J7']) ? { animationDuration: '0.7s' } : {}} />

          {/* Zone A header: Pre-treatment (M1, M2) */}
          <line x1="80" y1="55" x2="80" y2="80"
            className={isFlowing(flows['J2']) ? 'pipe-flow' : 'pipe-idle'} />
          <line x1="80" y1="80" x2="280" y2="80"
            className={isFlowing(flows['J2']) ? 'pipe-flow' : 'pipe-idle'}
            style={isFlowing(flows['J2']) ? { animationDuration: '0.8s' } : {}} />
          {/* Drop to M1 */}
          <line x1="100" y1="80" x2="100" y2={100 - 20}
            className={isFlowing(flows['J5']) ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M2 */}
          <line x1="260" y1="80" x2="260" y2={100 - 20}
            className={isFlowing(flows['J6']) ? 'pipe-flow' : 'pipe-idle'} />

          {/* Zone B header: Dyeing & Washing (M3, M4, M5) */}
          <line x1="80" y1="55" x2="80" y2="210"
            className={isFlowing(flows['J3']) ? 'pipe-flow' : 'pipe-idle'} />
          <line x1="80" y1="210" x2="440" y2="210"
            className={isFlowing(flows['J3']) ? 'pipe-flow' : 'pipe-idle'}
            style={isFlowing(flows['J3']) ? { animationDuration: '0.8s' } : {}} />
          {/* Drop to M3 */}
          <line x1="100" y1="210" x2="100" y2={230 - 20}
            className={isFlowing(flows['J8']) ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M4 */}
          <line x1="260" y1="210" x2="260" y2={230 - 20}
            className={isFlowing(flows['J9']) ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M5 */}
          <line x1="420" y1="210" x2="420" y2={230 - 20}
            className={isFlowing(flows['J10']) ? 'pipe-flow' : 'pipe-idle'} />

          {/* Zone C header: Finishing & Utility (M6, M7, M8) */}
          <line x1="80" y1="55" x2="80" y2="340"
            className={isFlowing(flows['J4']) || isFlowing(flows['J13']) ? 'pipe-flow' : 'pipe-idle'} />
          <line x1="80" y1="340" x2="440" y2="340"
            className={isFlowing(flows['J4']) || isFlowing(flows['J13']) ? 'pipe-flow' : 'pipe-idle'}
            style={isFlowing(flows['J4']) ? { animationDuration: '0.8s' } : {}} />
          {/* Drop to M6 */}
          <line x1="100" y1="340" x2="100" y2={360 - 20}
            className={isFlowing(flows['J11']) ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M7 */}
          <line x1="260" y1="340" x2="260" y2={360 - 20}
            className={isFlowing(flows['J12']) ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M8 */}
          <line x1="420" y1="340" x2="420" y2={360 - 20}
            className={isFlowing(flows['J13']) ? 'pipe-flow' : 'pipe-idle'} />

          {/* Tap branch pipes */}
          {/* T1 */}
          <line x1="580" y1="130" x2="565" y2="130"
            className={isFlowing(flows['J14']) ? 'pipe-flow' : 'pipe-idle'} />
          {/* T2 */}
          <line x1="580" y1="260" x2="565" y2="260"
            className={isFlowing(flows['J15']) ? 'pipe-flow' : 'pipe-idle'} />
          {/* T3 */}
          <line x1="580" y1="390" x2="565" y2="390"
            className={isFlowing(flows['J16']) ? 'pipe-flow' : 'pipe-idle'} />


          {/* Machine nodes */}
          {Object.entries(LAYOUT).filter(([id]) => id.startsWith('M')).map(([id, pos]) => {
            const m = machines[id] || { state: 'OFF', flow_lpm: 0, production_pct: 0 };
            const isRunning = m.state === 'RUNNING';
            const jid = JUNCTION_MAP[id];
            return (
              <g key={id} style={{ cursor: 'pointer' }} onClick={() => {
                setSelected(id);
                setControlState({ pct: m.production_pct ?? 100, state: m.state || 'OFF' });
                placePanel(pos);
              }}>
                <rect x={pos.x - 28} y={pos.y - 18} width={56} height={36} rx={4}
                  className={`node-machine ${isRunning ? 'running' : ''} ${leaking.has(JUNCTION_MAP[id]) ? 'node-leaking' : ''}`} />
                {leaking.has(JUNCTION_MAP[id]) && (
                  <rect x={pos.x - 34} y={pos.y - 24} width={68} height={48} rx={6}
                    className="leak-halo" />
                )}
                <text x={pos.x} y={pos.y - 4} textAnchor="middle" fontSize="8" fontWeight="700" fill="var(--text-primary)">{id}</text>
                <text x={pos.x} y={pos.y + 8} textAnchor="middle" fontSize="7" fill="var(--text-muted)">{pos.label}</text>
                <text x={pos.x} y={pos.y + 26} textAnchor="middle" fontSize="7" fontWeight="600"
                  fill={isRunning ? 'var(--accent)' : 'var(--text-muted)'}>
                  {m.flow_lpm?.toFixed(0) || 0} L/m
                </text>
              </g>
            );
          })}

          {/* Tap nodes */}
          {Object.entries(LAYOUT).filter(([id]) => id.startsWith('T')).map(([id, pos]) => {
            const t = taps[id] || { state: 'CLOSED', flow_lpm: 0 };
            const isOpen = t.state === 'OPEN';
            const points = `${pos.x},${pos.y - 14} ${pos.x - 16},${pos.y + 12} ${pos.x + 16},${pos.y + 12}`;
            return (
              <g key={id} style={{ cursor: 'pointer' }} onClick={() => { setSelected(id); placePanel(pos); }}>
                <polygon points={points}
                  className={`node-tap ${isOpen ? 'open' : ''} ${leaking.has(JUNCTION_MAP[id]) ? 'node-leaking' : ''}`} />
                <text x={pos.x} y={pos.y + 4} textAnchor="middle" fontSize="8" fontWeight="700" fill="var(--text-primary)">{id}</text>
                <text x={pos.x} y={pos.y + 28} textAnchor="middle" fontSize="7" fontWeight="600"
                  fill={isOpen ? 'var(--accent)' : 'var(--text-muted)'}>
                  {t.flow_lpm?.toFixed(0) || 0} L/m
                </text>
              </g>
            );
          })}
        </svg>

        {/* Inline control panel */}
        {selected && selected.startsWith('M') && (
          <div className="inline-control" style={{
            top: panelPos.top, left: panelPos.left, position: 'absolute',
          }}>
            <label>{selected} — Control</label>
            <label style={{ marginTop: 8 }}>Production: {controlState.pct}%</label>
            <input type="range" min="0" max="200" value={controlState.pct}
              onChange={e => setControlState(s => ({ ...s, pct: +e.target.value }))} />
            <label>State</label>
            <select value={controlState.state}
              onChange={e => setControlState(s => ({ ...s, state: e.target.value }))}>
              {MACHINE_STATES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <button className="apply-btn" disabled={busy}
              onClick={() => handleApplyMachine(selected)}>{busy ? 'Applying…' : 'Apply'}</button>
            <button className="apply-btn" style={{ marginTop: 4, background: 'var(--bg-inset)', color: 'var(--text-secondary)' }}
              onClick={() => setSelected(null)}>Cancel</button>
          </div>
        )}

        {selected && selected.startsWith('T') && (
          <div className="inline-control" style={{
            top: panelPos.top, left: panelPos.left, position: 'absolute',
          }}>
            <label>{selected} — Control</label>
            <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
              <button className="apply-btn" disabled={busy}
                onClick={() => handleApplyTap(selected, 'OPEN')}>Open</button>
              <button className="apply-btn" style={{ background: 'var(--bg-inset)', color: 'var(--text-secondary)' }} disabled={busy}
                onClick={() => handleApplyTap(selected, 'CLOSED')}>Close</button>
            </div>
            <button className="apply-btn" style={{ marginTop: 4, background: 'var(--bg-inset)', color: 'var(--text-secondary)' }}
              onClick={() => setSelected(null)}>Cancel</button>
          </div>
        )}
      </div>
    </div>
  );
}
