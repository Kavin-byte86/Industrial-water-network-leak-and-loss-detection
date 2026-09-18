/**
 * View 4 — Historical Data
 *
 * Line chart of flow over time with junction/machine selector.
 * Raw data table below matching the flat datasink schema.
 * All data fetched from GET /state/history — no fabricated content.
 */

import { useState, useRef, useEffect } from 'react';
import { useHistory } from '../../hooks/useHistory';

const JUNCTIONS = Array.from({ length: 16 }, (_, i) => `J${i + 1}`);
const ALL_SERIES = JUNCTIONS;

const COLORS = [
  '#2b6cb0', '#e53e3e', '#38a169', '#d69e2e', '#805ad5',
  '#dd6b20', '#3182ce', '#e53e3e', '#319795', '#d53f8c',
  '#2c5282', '#c53030', '#276749', '#b7791f', '#6b46c1',
  '#c05621',
];

function SimpleLineChart({ history, selectedSeries }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !history.length || !selectedSeries.length) return;

    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.offsetWidth * 2;
    const H = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    const w = W / 2, h = H / 2;

    ctx.clearRect(0, 0, w, h);

    const pad = { top: 20, right: 16, bottom: 30, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Find global min/max across selected series
    let allVals = [];
    for (const sid of selectedSeries) {
      for (const tick of history) {
        allVals.push(tick.flows?.[sid] || 0);
      }
    }
    const minVal = Math.min(0, ...allVals);
    const maxVal = Math.max(1, ...allVals);
    const range = maxVal - minVal || 1;

    // Grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      const val = maxVal - (range * i) / 4;
      ctx.fillStyle = '#718096';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(0), pad.left - 6, y + 3);
    }

    // Axis labels
    ctx.fillStyle = '#4a5568';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('L/min', pad.left - 10, pad.top - 8);

    // X-axis: show tick indices
    const step = Math.max(1, Math.floor(history.length / 8));
    for (let i = 0; i < history.length; i += step) {
      const x = pad.left + (i / (history.length - 1 || 1)) * plotW;
      ctx.fillStyle = '#718096';
      ctx.fillText(`t-${history.length - i}`, x, h - pad.bottom + 16);
    }

    // Draw each series
    selectedSeries.forEach((sid, si) => {
      ctx.strokeStyle = COLORS[JUNCTIONS.indexOf(sid) % COLORS.length];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      history.forEach((tick, i) => {
        const val = tick.flows?.[sid] || 0;
        const x = pad.left + (i / (history.length - 1 || 1)) * plotW;
        const y = pad.top + plotH - ((val - minVal) / range) * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }, [history, selectedSeries]);

  return (
    <canvas ref={canvasRef} className="chart-canvas"
      style={{ width: '100%', height: '300px' }} />
  );
}

export function HistoryView() {
  const { history, count } = useHistory(200);
  const [selectedSeries, setSelectedSeries] = useState(['J1']);
  const [showRawRows, setShowRawRows] = useState(20);

  const toggleSeries = (sid) => {
    setSelectedSeries(prev =>
      prev.includes(sid) ? prev.filter(s => s !== sid) : [...prev, sid]
    );
  };

  return (
    <div>
      <div className="view-header">
        <h2 className="view-title">Historical Data</h2>
        <span style={{ fontSize: '0.75rem', color: '#718096' }}>
          {count} ticks loaded
        </span>
      </div>

      {/* Junction selector */}
      <div className="card">
        <div className="card-header">Flow Chart — Select Junctions</div>
        <div className="chart-selector">
          {ALL_SERIES.map(sid => (
            <button key={sid}
              className={selectedSeries.includes(sid) ? 'selected' : ''}
              onClick={() => toggleSeries(sid)}>
              {sid}
            </button>
          ))}
        </div>
        <SimpleLineChart history={history} selectedSeries={selectedSeries} />
      </div>

      {/* Raw data table */}
      <div className="card">
        <div className="card-header">
          Raw Tick Data
          <span style={{ fontSize: '0.7rem', color: '#718096', marginLeft: 'auto' }}>
            Showing last {Math.min(showRawRows, history.length)} of {count}
          </span>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Shift</th>
                {JUNCTIONS.map(j => <th key={j}>{j}</th>)}
              </tr>
            </thead>
            <tbody>
              {history.slice(-showRawRows).reverse().map((tick, i) => (
                <tr key={i}>
                  <td style={{ whiteSpace: 'nowrap' }}>{tick.timestamp?.replace('T', ' ')}</td>
                  <td>{tick.shift}</td>
                  {JUNCTIONS.map(j => (
                    <td key={j}>{(tick.flows?.[j] || 0).toFixed(1)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {history.length > showRawRows && (
          <button style={{ marginTop: 8, fontSize: '0.75rem', color: '#3182ce', background: 'none',
            border: 'none', cursor: 'pointer', fontFamily: 'var(--font-family)' }}
            onClick={() => setShowRawRows(r => r + 20)}>
            Show more rows…
          </button>
        )}
      </div>
    </div>
  );
}
