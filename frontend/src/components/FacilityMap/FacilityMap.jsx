/**
 * View 2 — Facility Map (2D top-down factory floor layout)
 *
 * Spatial layout resembling an actual textile mill floor.
 * Machines placed in production zones, taps at utility areas.
 * Click a machine/tap to open inline control panel.
 */

import { useState } from 'react';
import { api } from '../../api/client';

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

export function FacilityMap({ state }) {
  const [selected, setSelected] = useState(null);
  const [controlState, setControlState] = useState({ pct: 100, state: 'RUNNING' });
  const flows = state?.flows || {};
  const machines = state?.machines || {};
  const taps = state?.taps || {};

  const handleApplyMachine = async (mid) => {
    await api.controlMachine(mid, controlState.pct, controlState.state);
    setSelected(null);
  };

  const handleApplyTap = async (tid, newState) => {
    await api.controlTap(tid, newState);
    setSelected(null);
  };

  return (
    <div>
      <div className="view-header">
        <h2 className="view-title">Facility Map</h2>
      </div>
      <div className="facility-map" style={{ position: 'relative' }}>
        <svg viewBox="0 0 700 460" preserveAspectRatio="xMidYMid meet"
          style={{ width: '100%', height: 'calc(100vh - 100px)', background: '#fff', border: '1px solid var(--gray-300)', borderRadius: '8px' }}>

          {/* Background zones */}
          <rect x="60" y="60" width="370" height="90" rx="6" fill="#ebf8ff" stroke="#bee3f8" strokeDasharray="4" />
          <text x="245" y="78" textAnchor="middle" fontSize="8" fill="#4299e1" fontWeight="600">PRE-TREATMENT</text>

          <rect x="60" y="190" width="430" height="90" rx="6" fill="#ebf8ff" stroke="#bee3f8" strokeDasharray="4" />
          <text x="275" y="208" textAnchor="middle" fontSize="8" fill="#4299e1" fontWeight="600">DYEING & WASHING</text>

          <rect x="60" y="320" width="430" height="90" rx="6" fill="#ebf8ff" stroke="#bee3f8" strokeDasharray="4" />
          <text x="275" y="338" textAnchor="middle" fontSize="8" fill="#4299e1" fontWeight="600">FINISHING & UTILITY</text>

          <rect x="540" y="90" width="120" height="340" rx="6" fill="#f0fff4" stroke="#c6f6d5" strokeDasharray="4" />
          <text x="600" y="108" textAnchor="middle" fontSize="8" fill="#48bb78" fontWeight="600">TAPS</text>

          {/* ═══ PIPE NETWORK ═══ */}
          {/* Main supply line from inlet */}
          <text x="350" y="22" textAnchor="middle" fontSize="9" fontWeight="700" fill="#2b6cb0">
            MAIN INLET (J1): {(flows['J1'] || 0).toFixed(0)} L/min
          </text>
          <line x1="350" y1="30" x2="350" y2="55" className="pipe-flow" style={{ animationDuration: '0.8s' }} />

          {/* Main horizontal trunk line */}
          <line x1="70" y1="55" x2="620" y2="55"
            className={flows['J1'] > 0 ? 'pipe-flow' : 'pipe-idle'}
            style={flows['J1'] > 0 ? { animationDuration: '0.6s', strokeWidth: 3 } : { strokeWidth: 2 }} />

          {/* Branch to taps zone */}
          <line x1="580" y1="55" x2="580" y2="410"
            className={flows['J7'] > 0 ? 'pipe-flow' : 'pipe-idle'}
            style={flows['J7'] > 0 ? { animationDuration: '0.7s' } : {}} />

          {/* Zone A header: Pre-treatment (M1, M2) */}
          <line x1="80" y1="55" x2="80" y2="80"
            className={flows['J2'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          <line x1="80" y1="80" x2="280" y2="80"
            className={flows['J2'] > 0 ? 'pipe-flow' : 'pipe-idle'}
            style={flows['J2'] > 0 ? { animationDuration: '0.8s' } : {}} />
          {/* Drop to M1 */}
          <line x1="100" y1="80" x2="100" y2={100 - 20}
            className={flows['J5'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M2 */}
          <line x1="260" y1="80" x2="260" y2={100 - 20}
            className={flows['J6'] > 0 ? 'pipe-flow' : 'pipe-idle'} />

          {/* Zone B header: Dyeing & Washing (M3, M4, M5) */}
          <line x1="80" y1="55" x2="80" y2="210"
            className={flows['J3'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          <line x1="80" y1="210" x2="440" y2="210"
            className={flows['J3'] > 0 ? 'pipe-flow' : 'pipe-idle'}
            style={flows['J3'] > 0 ? { animationDuration: '0.8s' } : {}} />
          {/* Drop to M3 */}
          <line x1="100" y1="210" x2="100" y2={230 - 20}
            className={flows['J8'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M4 */}
          <line x1="260" y1="210" x2="260" y2={230 - 20}
            className={flows['J9'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M5 */}
          <line x1="420" y1="210" x2="420" y2={230 - 20}
            className={flows['J10'] > 0 ? 'pipe-flow' : 'pipe-idle'} />

          {/* Zone C header: Finishing & Utility (M6, M7, M8) */}
          <line x1="80" y1="55" x2="80" y2="340"
            className={flows['J4'] > 0 || flows['J13'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          <line x1="80" y1="340" x2="440" y2="340"
            className={flows['J4'] > 0 || flows['J13'] > 0 ? 'pipe-flow' : 'pipe-idle'}
            style={flows['J4'] > 0 ? { animationDuration: '0.8s' } : {}} />
          {/* Drop to M6 */}
          <line x1="100" y1="340" x2="100" y2={360 - 20}
            className={flows['J11'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M7 */}
          <line x1="260" y1="340" x2="260" y2={360 - 20}
            className={flows['J12'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          {/* Drop to M8 */}
          <line x1="420" y1="340" x2="420" y2={360 - 20}
            className={flows['J13'] > 0 ? 'pipe-flow' : 'pipe-idle'} />

          {/* Tap branch pipes */}
          {/* T1 */}
          <line x1="580" y1="130" x2="565" y2="130"
            className={flows['J14'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          {/* T2 */}
          <line x1="580" y1="260" x2="565" y2="260"
            className={flows['J15'] > 0 ? 'pipe-flow' : 'pipe-idle'} />
          {/* T3 */}
          <line x1="580" y1="390" x2="565" y2="390"
            className={flows['J16'] > 0 ? 'pipe-flow' : 'pipe-idle'} />


          {/* Machine nodes */}
          {Object.entries(LAYOUT).filter(([id]) => id.startsWith('M')).map(([id, pos]) => {
            const m = machines[id] || { state: 'OFF', flow_lpm: 0, production_pct: 0 };
            const isRunning = m.state === 'RUNNING';
            const jid = JUNCTION_MAP[id];
            return (
              <g key={id} style={{ cursor: 'pointer' }} onClick={() => {
                setSelected(id);
                setControlState({ pct: m.production_pct || 100, state: m.state || 'OFF' });
              }}>
                <rect x={pos.x - 28} y={pos.y - 18} width={56} height={36} rx={4}
                  className={`node-machine ${isRunning ? 'running' : ''}`} />
                <text x={pos.x} y={pos.y - 4} textAnchor="middle" fontSize="8" fontWeight="700" fill="#2d3748">{id}</text>
                <text x={pos.x} y={pos.y + 8} textAnchor="middle" fontSize="7" fill="#718096">{pos.label}</text>
                <text x={pos.x} y={pos.y + 26} textAnchor="middle" fontSize="7" fontWeight="600"
                  fill={isRunning ? '#2b6cb0' : '#a0aec0'}>
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
              <g key={id} style={{ cursor: 'pointer' }} onClick={() => setSelected(id)}>
                <polygon points={points} className={`node-tap ${isOpen ? 'open' : ''}`} />
                <text x={pos.x} y={pos.y + 4} textAnchor="middle" fontSize="8" fontWeight="700" fill="#2d3748">{id}</text>
                <text x={pos.x} y={pos.y + 28} textAnchor="middle" fontSize="7" fontWeight="600"
                  fill={isOpen ? '#2b6cb0' : '#a0aec0'}>
                  {t.flow_lpm?.toFixed(0) || 0} L/m
                </text>
              </g>
            );
          })}
        </svg>

        {/* Inline control panel */}
        {selected && selected.startsWith('M') && (
          <div className="inline-control" style={{
            top: LAYOUT[selected].y + 60, left: LAYOUT[selected].x + 30,
            position: 'absolute',
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
            <button className="apply-btn" onClick={() => handleApplyMachine(selected)}>Apply</button>
            <button className="apply-btn" style={{ marginTop: 4, background: '#a0aec0' }}
              onClick={() => setSelected(null)}>Cancel</button>
          </div>
        )}

        {selected && selected.startsWith('T') && (
          <div className="inline-control" style={{
            top: LAYOUT[selected].y + 60, left: LAYOUT[selected].x + 30,
            position: 'absolute',
          }}>
            <label>{selected} — Control</label>
            <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
              <button className="apply-btn" onClick={() => handleApplyTap(selected, 'OPEN')}>Open</button>
              <button className="apply-btn" style={{ background: '#a0aec0' }}
                onClick={() => handleApplyTap(selected, 'CLOSED')}>Close</button>
            </div>
            <button className="apply-btn" style={{ marginTop: 4, background: '#e2e8f0', color: '#4a5568' }}
              onClick={() => setSelected(null)}>Cancel</button>
          </div>
        )}
      </div>
    </div>
  );
}
