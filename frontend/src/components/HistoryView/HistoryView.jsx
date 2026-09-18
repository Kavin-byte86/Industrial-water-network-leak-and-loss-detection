/**
 * View 4 — Historical Data
 *
 * Line chart of flow over time with a junction selector, plus the raw tick
 * table. All data comes from GET /state/history — nothing is fabricated.
 *
 * Colour rules: hues are assigned in fixed order from a validated categorical
 * palette and are held by the series that claimed them, so deselecting one
 * junction never repaints the others. Concurrent series are capped at the
 * palette size rather than cycling hues.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { useHistory } from '../../hooks/useHistory';

const JUNCTIONS = Array.from({ length: 16 }, (_, i) => `J${i + 1}`);

// Validated against the dark chart surface (#151d2e): lightness band, chroma
// floor, CVD separation, normal-vision floor and >=3:1 contrast all pass.
const SERIES_COLORS = [
  '#3987e5', '#d95926', '#199e70', '#c98500',
  '#d55181', '#008300', '#9085e9', '#e66767',
];
const MAX_SERIES = SERIES_COLORS.length;

const AXIS_INK = '#6b7f9e';
const GRID_INK = 'rgba(255,255,255,0.07)';
const SURFACE = '#151d2e';

function SimpleLineChart({ history, series, colorOf }) {
  const canvasRef = useRef(null);
  const geomRef = useRef(null);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Clear first, bail out second — returning before the clear used to leave
    // the previous chart painted after every series was deselected.
    if (!history.length || !series.length) {
      geomRef.current = null;
      return;
    }

    const pad = { top: 18, right: 18, bottom: 28, left: 54 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    let maxVal = 1;
    for (const sid of series) {
      for (const tick of history) {
        const v = tick.flows?.[sid] || 0;
        if (v > maxVal) maxVal = v;
      }
    }
    const range = maxVal || 1;
    const xAt = i => pad.left + (i / (history.length - 1 || 1)) * plotW;
    const yAt = v => pad.top + plotH - (v / range) * plotH;

    geomRef.current = { pad, plotW, count: history.length };

    // Recessive grid and axis ink
    ctx.strokeStyle = GRID_INK;
    ctx.lineWidth = 1;
    ctx.fillStyle = AXIS_INK;
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText((maxVal - (range * i) / 4).toFixed(0), pad.left - 8, y + 3);
    }

    ctx.textAlign = 'left';
    ctx.fillText('L/min', 4, pad.top - 5);

    // A few x labels, never one per tick
    ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(history.length / 6));
    for (let i = 0; i < history.length; i += step) {
      ctx.fillText(`t-${history.length - i}`, xAt(i), h - pad.bottom + 16);
    }

    // Series: thin 2px marks
    series.forEach(sid => {
      ctx.strokeStyle = colorOf(sid);
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      history.forEach((tick, i) => {
        const y = yAt(tick.flows?.[sid] || 0);
        if (i === 0) ctx.moveTo(xAt(i), y); else ctx.lineTo(xAt(i), y);
      });
      ctx.stroke();
    });

    // Crosshair on the hovered tick
    if (hover != null && hover < history.length) {
      const x = xAt(hover);
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();

      series.forEach(sid => {
        const y = yAt(history[hover].flows?.[sid] || 0);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = colorOf(sid);
        ctx.fill();
        // 2px surface ring keeps overlapping markers separable
        ctx.strokeStyle = SURFACE;
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }
  }, [history, series, hover, colorOf]);

  const onMove = useCallback(e => {
    const geom = geomRef.current;
    const canvas = canvasRef.current;
    if (!geom || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const frac = (e.clientX - rect.left - geom.pad.left) / (geom.plotW || 1);
    const idx = Math.round(frac * (geom.count - 1));
    setHover(idx >= 0 && idx < geom.count ? idx : null);
  }, []);

  const tick = hover != null ? history[hover] : null;

  return (
    <div className="chart-wrap">
      <canvas
        ref={canvasRef}
        className="chart-canvas"
        style={{ width: '100%', height: '320px' }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      />
      {tick && (
        <div className="chart-tooltip">
          <div className="chart-tooltip-time">{tick.timestamp?.replace('T', ' ')}</div>
          {series.map(sid => (
            <div key={sid} className="chart-tooltip-row">
              <i style={{ background: colorOf(sid) }} />
              <span>{sid}</span>
              <strong>{(tick.flows?.[sid] ?? 0).toFixed(1)}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function HistoryView() {
  const { history, count, error } = useHistory(200);

  // sid -> palette slot. A slot is held for as long as the series is selected,
  // so removing one series never recolours the others.
  const [slots, setSlots] = useState({ J1: 0 });
  const [showRawRows, setShowRawRows] = useState(20);

  const series = Object.keys(slots);
  const atCapacity = series.length >= MAX_SERIES;

  const colorOf = useCallback(sid => SERIES_COLORS[slots[sid] ?? 0], [slots]);

  const toggleSeries = (sid) => {
    setSlots(prev => {
      if (sid in prev) {
        const next = { ...prev };
        delete next[sid];
        return next;
      }
      const used = new Set(Object.values(prev));
      let slot = 0;
      while (used.has(slot) && slot < MAX_SERIES) slot++;
      if (slot >= MAX_SERIES) return prev;   // at capacity — ignore
      return { ...prev, [sid]: slot };
    });
  };

  const muted = {
    fontSize: '0.7rem', color: 'var(--text-muted)', marginLeft: 'auto',
    textTransform: 'none', letterSpacing: 0, fontWeight: 500,
  };

  return (
    <div>
      <div className="view-header">
        <h2 className="view-title">Historical Data</h2>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {count} ticks loaded
        </span>
      </div>

      <div className="card">
        <div className="card-header">
          Flow over time
          <span style={muted}>{series.length}/{MAX_SERIES} series</span>
        </div>

        <div className="chart-selector">
          {JUNCTIONS.map(sid => {
            const on = sid in slots;
            return (
              <button key={sid}
                className={on ? 'selected' : ''}
                disabled={!on && atCapacity}
                title={!on && atCapacity ? `Deselect a series to add ${sid}` : sid}
                onClick={() => toggleSeries(sid)}>
                {sid}
              </button>
            );
          })}
        </div>

        {series.length > 0 && (
          <div className="chart-legend">
            {series.map(sid => (
              <span key={sid}>
                <i style={{ background: colorOf(sid) }} />
                {sid}
              </span>
            ))}
          </div>
        )}

        {series.length === 0 && (
          <p className="chart-empty">Select at least one junction to plot.</p>
        )}
        {series.length > 0 && history.length === 0 && (
          <p className="chart-empty">
            {error ? `History unavailable: ${error}` : 'Waiting for tick history…'}
          </p>
        )}

        <SimpleLineChart history={history} series={series} colorOf={colorOf} />
      </div>

      <div className="card">
        <div className="card-header">
          Raw tick data
          <span style={muted}>
            last {Math.min(showRawRows, history.length)} of {count}
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
          <button className="link-btn" onClick={() => setShowRawRows(r => r + 20)}>
            Show more rows…
          </button>
        )}
      </div>
    </div>
  );
}
