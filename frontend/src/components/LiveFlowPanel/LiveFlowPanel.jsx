/**
 * View 3 — Live Flow Data
 *
 * Data-dense table of every current sensor reading:
 * 16 junction flows, 8 machine states, 3 tap states, pressures.
 * Inline sparkline trends from rolling buffer.
 */

const JUNCTIONS = Array.from({ length: 16 }, (_, i) => `J${i + 1}`);
const MACHINES = Array.from({ length: 8 }, (_, i) => `M${i + 1}`);
const TAPS = Array.from({ length: 3 }, (_, i) => `T${i + 1}`);

function Sparkline({ values, width = 60, height = 16 }) {
  if (!values || values.length < 2) return null;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const step = width / (values.length - 1);
  const points = values.map((v, i) =>
    `${i * step},${height - ((v - min) / range) * height}`
  ).join(' ');

  return (
    <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline points={points} fill="none" stroke="#4299e1" strokeWidth="1.2" />
    </svg>
  );
}

function StateBadge({ state }) {
  const cls = (state || 'OFF').toLowerCase().replace(/\s+/g, '');
  return <span className={`badge badge-${cls}`}>{state || 'OFF'}</span>;
}

export function LiveFlowPanel({ state, buffer }) {
  const flows = state?.flows || {};
  const pressures = state?.pressures || {};
  const machines = state?.machines || {};
  const taps = state?.taps || {};

  // Extract sparkline data from buffer
  const getFlowHistory = (jid) => buffer.map(s => s.flows?.[jid] || 0);
  const getPressureHistory = (jid) => buffer.map(s => s.pressures?.[jid] || 0);

  return (
    <div>
      <div className="view-header">
        <h2 className="view-title">Live Flow Data</h2>
        <span style={{ fontSize: '0.75rem', color: '#718096' }}>
          {buffer.length} ticks buffered • Updates every poll interval
        </span>
      </div>

      {/* Junction Flows & Pressures */}
      <div className="card">
        <div className="card-header">Junction Sensors (16)</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Junction</th>
              <th>Flow (L/min)</th>
              <th>Trend</th>
              <th>Pressure (bar)</th>
              <th>P. Trend</th>
            </tr>
          </thead>
          <tbody>
            {JUNCTIONS.map(jid => (
              <tr key={jid}>
                <td style={{ fontWeight: 600 }}>{jid}</td>
                <td>{(flows[jid] || 0).toFixed(1)}</td>
                <td><Sparkline values={getFlowHistory(jid)} /></td>
                <td>{(pressures[jid] || 0).toFixed(3)}</td>
                <td><Sparkline values={getPressureHistory(jid)} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Machine States */}
      <div className="card">
        <div className="card-header">Machines (8)</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Machine</th>
              <th>State</th>
              <th>Production %</th>
              <th>Flow (L/min)</th>
              <th>Trend</th>
            </tr>
          </thead>
          <tbody>
            {MACHINES.map(mid => {
              const m = machines[mid] || {};
              const jid = {M1:'J5',M2:'J6',M3:'J8',M4:'J9',M5:'J10',M6:'J11',M7:'J12',M8:'J13'}[mid];
              return (
                <tr key={mid}>
                  <td style={{ fontWeight: 600 }}>{mid}</td>
                  <td><StateBadge state={m.state} /></td>
                  <td>{(m.production_pct || 0).toFixed(0)}%</td>
                  <td>{(m.flow_lpm || 0).toFixed(1)}</td>
                  <td><Sparkline values={getFlowHistory(jid)} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Tap States */}
      <div className="card">
        <div className="card-header">Taps (3)</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Tap</th>
              <th>State</th>
              <th>Flow (L/min)</th>
              <th>Trend</th>
            </tr>
          </thead>
          <tbody>
            {TAPS.map(tid => {
              const t = taps[tid] || {};
              const jid = {T1:'J14',T2:'J15',T3:'J16'}[tid];
              return (
                <tr key={tid}>
                  <td style={{ fontWeight: 600 }}>{tid}</td>
                  <td><StateBadge state={t.state} /></td>
                  <td>{(t.flow_lpm || 0).toFixed(1)}</td>
                  <td><Sparkline values={getFlowHistory(jid)} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
