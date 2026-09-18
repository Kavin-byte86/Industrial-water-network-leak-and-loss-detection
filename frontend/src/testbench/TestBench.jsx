/**
 * Test Bench — leak simulation console.
 *
 * A separate page from the operator dashboard. Everything here writes to the
 * simulation: inject a leak at a node, force a raw sensor reading, or drive
 * machines and taps directly. The dashboard then has to detect what was done.
 *
 * The panel on the right shows the detector's verdict next to the ground truth,
 * so it is immediately obvious whether a given leak was actually caught.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';

const JUNCTIONS = Array.from({ length: 16 }, (_, i) => `J${i + 1}`);
const MACHINES = Array.from({ length: 8 }, (_, i) => `M${i + 1}`);
const TAPS = Array.from({ length: 3 }, (_, i) => `T${i + 1}`);
const MACHINE_STATES = ['OFF', 'STARTING', 'RUNNING', 'STOPPING', 'MAINTENANCE'];

// Which machine/tap each leaf junction feeds — for labelling only.
const JUNCTION_FEEDS = {
  J5: 'M1', J6: 'M2', J8: 'M3', J9: 'M4', J10: 'M5', J11: 'M6',
  J12: 'M7', J13: 'M8', J14: 'T1', J15: 'T2', J16: 'T3',
};

const JUNCTION_ROLE = {
  J1: 'Main inlet header',
  J2: 'Branch A manifold',
  J3: 'Branch B manifold',
  J4: 'Branch C manifold',
  J7: 'Branch D utility manifold',
};

const junctionLabel = (jid) =>
  JUNCTION_ROLE[jid] || `Feed line to ${JUNCTION_FEEDS[jid] || '—'}`;

const POLL_MS = 2000;

export default function TestBench() {
  const [snap, setSnap] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  // Leak form
  const [leakNode, setLeakNode] = useState('J3');
  const [leakRate, setLeakRate] = useState(120);

  // Override form
  const [ovKind, setOvKind] = useState('flow');
  const [ovNode, setOvNode] = useState('J1');
  const [ovValue, setOvValue] = useState(0);

  // Read inside `run` without making it depend on each poll.
  const pausedRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getTestbenchSnapshot();
      pausedRef.current = Boolean(data.paused);
      setSnap(data);
      setConnected(true);
      setError(null);
    } catch (err) {
      setConnected(false);
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  // Every write goes through here so a failure is always visible rather than
  // becoming an unhandled promise rejection.
  //
  // `advance` steps the clock once after the write. Flows are only recomputed
  // on a tick, so while the simulation is paused an injected leak would sit in
  // the registry with nothing on screen changing — which reads as "the button
  // did nothing". Stepping makes every action take effect immediately.
  const run = useCallback(async (action, message, { advance = true } = {}) => {
    try {
      await action();
      let suffix = '';
      if (advance && pausedRef.current) {
        await api.step();
        suffix = ' · stepped 1 tick';
      }
      setError(null);
      setNotice(message + suffix);
      await refresh();
      setTimeout(() => setNotice(null), 3000);
    } catch (err) {
      setError(err.message);
    }
  }, [refresh]);

  const tick = snap?.tick;
  const detection = tick?.detection;
  const flows = tick?.flows || {};
  const pressures = tick?.pressures || {};
  const machines = tick?.machines || {};
  const taps = tick?.taps || {};
  const leaks = snap?.leaks || {};
  const overrides = snap?.overrides || {};

  const injectedTotal = useMemo(
    () => Object.values(leaks).reduce((a, b) => a + b, 0),
    [leaks]
  );

  // Ground truth vs detector — the point of this page. Compared as sets, so
  // injecting two leaks and catching only one reads as "partial", not "correct".
  const truthNodes = Object.keys(leaks);
  const detectedNodes = detection?.leaks?.length
    ? detection.leaks.map(l => l.node)
    : detection?.leak_detected ? [detection.leak_node] : [];

  const verdict = (() => {
    if (!detection) return null;
    const truth = new Set(truthNodes);
    const found = new Set(detectedNodes);
    const hit = truthNodes.filter(n => found.has(n));
    const missed = truthNodes.filter(n => !found.has(n));
    const spurious = detectedNodes.filter(n => !truth.has(n));

    if (!truth.size && !found.size) {
      return { kind: 'ok', text: 'Clean — no leak injected, none detected.' };
    }
    if (!truth.size) {
      return { kind: 'bad', text: `False alarm — detector flagged ${spurious.join(', ')} with nothing injected.` };
    }
    if (!found.size) {
      return { kind: 'bad', text: `Missed — ${truthNodes.join(', ')} injected but nothing detected.` };
    }
    if (!missed.length && !spurious.length) {
      return {
        kind: 'ok',
        text: hit.length > 1
          ? `Correct — all ${hit.length} leaks localised (${hit.join(', ')}).`
          : `Correct — detected at ${hit[0]}, which is where the leak was injected.`,
      };
    }
    const parts = [];
    if (hit.length) parts.push(`found ${hit.join(', ')}`);
    if (missed.length) parts.push(`missed ${missed.join(', ')}`);
    if (spurious.length) parts.push(`false alarm at ${spurious.join(', ')}`);
    return { kind: 'warn', text: `Partial — ${parts.join('; ')}.` };
  })();

  return (
    <div className="tb-page">
      <header className="tb-header">
        <div>
          <h1>Leak Simulation Test Bench</h1>
          <p>
            Inject leaks and override sensors here; watch the operator dashboard
            react. This page is for testing only.
          </p>
        </div>
        <div className="tb-header-right">
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
          <span className="tb-conn">{connected ? 'Backend connected' : 'Backend offline'}</span>
          {snap?.paused && (
            <span className="tb-paused" title="The clock is stopped. Actions still apply — each one steps a single tick.">
              SIM PAUSED
            </span>
          )}
          <a className="tb-link" href="/" target="_blank" rel="noreferrer">
            Open dashboard ↗
          </a>
        </div>
      </header>

      {error && (
        <div className="app-banner app-banner-error">
          <strong>Request failed.</strong> {error}
        </div>
      )}
      {notice && <div className="app-banner app-banner-info">{notice}</div>}

      <div className="tb-grid">
        {/* ── Left column: the controls that cause a leak ── */}
        <section className="tb-col">
          <div className="card">
            <div className="card-header">Inject a leak</div>
            <div className="tb-form">
              <label>Location</label>
              <select value={leakNode} onChange={e => setLeakNode(e.target.value)}>
                {JUNCTIONS.map(j => (
                  <option key={j} value={j}>{j} — {junctionLabel(j)}</option>
                ))}
              </select>

              <label>Leak rate: {leakRate} L/min</label>
              <input type="range" min="0" max="600" step="5"
                value={leakRate}
                onChange={e => setLeakRate(Number(e.target.value))} />
              <input type="number" min="0" max="5000" value={leakRate}
                onChange={e => setLeakRate(Number(e.target.value))} />

              <p className="tb-hint">
                Water escapes below this junction's meter, so {leakNode} and
                everything upstream read higher while its downstream meters stay
                normal. Roughly 30 L/min is the smallest leak that clears sensor
                noise on the main.
              </p>

              <div className="tb-btn-row">
                <button className="tb-btn tb-btn-danger"
                  onClick={() => run(() => api.setLeak(leakNode, leakRate),
                    `Injected ${leakRate} L/min at ${leakNode}.`)}>
                  Inject leak
                </button>
                <button className="tb-btn"
                  onClick={() => run(() => api.clearAllLeaks(), 'All leaks cleared.')}>
                  Clear all
                </button>
              </div>
            </div>

            <table className="data-table tb-table">
              <thead>
                <tr><th>Active leak</th><th>Rate</th><th /></tr>
              </thead>
              <tbody>
                {Object.entries(leaks).length === 0 && (
                  <tr><td colSpan={3} className="tb-muted">No leaks injected.</td></tr>
                )}
                {Object.entries(leaks).map(([node, rate]) => (
                  <tr key={node}>
                    <td><strong>{node}</strong> <span className="tb-muted">{junctionLabel(node)}</span></td>
                    <td>{rate.toFixed(1)} L/min</td>
                    <td>
                      <button className="tb-btn tb-btn-sm"
                        onClick={() => run(() => api.clearLeak(node), `Cleared leak at ${node}.`)}>
                        Clear
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <div className="card-header">Override a raw sensor</div>
            <div className="tb-form">
              <p className="tb-hint">
                Forces one meter to report a fixed value, simulating a faulty or
                tampered sensor. Overrides are applied after the simulation, so
                the detector sees the forced number.
              </p>
              <label>Sensor</label>
              <div className="tb-inline">
                <select value={ovKind} onChange={e => setOvKind(e.target.value)}>
                  <option value="flow">flow (L/min)</option>
                  <option value="pressure">pressure (bar)</option>
                </select>
                <select value={ovNode} onChange={e => setOvNode(e.target.value)}>
                  {JUNCTIONS.map(j => <option key={j} value={j}>{j}</option>)}
                </select>
              </div>
              <label>Value</label>
              <input type="number" value={ovValue} step="0.1"
                onChange={e => setOvValue(Number(e.target.value))} />
              <div className="tb-btn-row">
                <button className="tb-btn"
                  onClick={() => run(
                    () => api.setOverride(`${ovKind}_${ovNode}`, ovValue),
                    `${ovKind}_${ovNode} forced to ${ovValue}.`)}>
                  Apply override
                </button>
                <button className="tb-btn"
                  onClick={() => run(() => api.clearAllOverrides(), 'All overrides cleared.')}>
                  Clear all
                </button>
              </div>
            </div>

            <table className="data-table tb-table">
              <thead><tr><th>Sensor</th><th>Forced value</th><th /></tr></thead>
              <tbody>
                {Object.entries(overrides).length === 0 && (
                  <tr><td colSpan={3} className="tb-muted">No overrides active.</td></tr>
                )}
                {Object.entries(overrides).map(([sensor, value]) => (
                  <tr key={sensor}>
                    <td>{sensor}</td>
                    <td>{value}</td>
                    <td>
                      <button className="tb-btn tb-btn-sm"
                        onClick={() => run(() => api.clearOverride(sensor), `Cleared ${sensor}.`)}>
                        Clear
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ── Middle column: plant inputs ── */}
        <section className="tb-col">
          <div className="card">
            <div className="card-header">Machines</div>
            <table className="data-table tb-table">
              <thead>
                <tr><th>ID</th><th>Production %</th><th>State</th><th>Flow</th></tr>
              </thead>
              <tbody>
                {MACHINES.map(mid => (
                  <MachineRow key={mid} mid={mid}
                    machine={machines[mid]}
                    onApply={(pct, st) => run(
                      () => api.setEndpoint(mid, pct, st),
                      `${mid} set to ${st} at ${pct}%.`)} />
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <div className="card-header">Taps</div>
            <table className="data-table tb-table">
              <thead><tr><th>ID</th><th>State</th><th>Flow</th><th /></tr></thead>
              <tbody>
                {TAPS.map(tid => {
                  const t = taps[tid] || {};
                  const open = t.state === 'OPEN';
                  return (
                    <tr key={tid}>
                      <td><strong>{tid}</strong></td>
                      <td><span className={`badge badge-${open ? 'open' : 'closed'}`}>{t.state || 'CLOSED'}</span></td>
                      <td>{(t.flow_lpm || 0).toFixed(1)}</td>
                      <td>
                        <button className="tb-btn tb-btn-sm"
                          onClick={() => run(
                            () => api.setEndpoint(tid, null, open ? 'CLOSED' : 'OPEN'),
                            `${tid} ${open ? 'closed' : 'opened'}.`)}>
                          {open ? 'Close' : 'Open'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="card">
            <div className="card-header">Simulation</div>
            <div className="tb-btn-row tb-pad">
              <button className="tb-btn" onClick={() => run(api.step, 'Stepped one tick.', { advance: false })}>Step</button>
              {snap?.paused
                ? <button className="tb-btn" onClick={() => run(api.resume, 'Resumed.', { advance: false })}>Resume</button>
                : <button className="tb-btn" onClick={() => run(api.pause, 'Paused.', { advance: false })}>Pause</button>}
              <button className="tb-btn" onClick={() => run(api.reset, 'Simulation reset.', { advance: false })}>Reset</button>
            </div>
            <p className="tb-hint tb-pad">
              Reset returns all machines to OFF and clears every leak and override.
            </p>
          </div>
        </section>

        {/* ── Right column: what the detector made of it ── */}
        <section className="tb-col">
          <div className={`card tb-verdict ${verdict ? `tb-verdict-${verdict.kind}` : ''}`}>
            <div className="card-header">Detector verdict</div>
            <div className="tb-pad">
              {!detection && <p className="tb-muted">Waiting for the first tick…</p>}
              {detection && (
                <>
                  <p className="tb-verdict-text">{verdict?.text}</p>
                  <dl className="tb-dl">
                    <dt>Ground truth</dt>
                    <dd>
                      {truthNodes.length
                        ? `${truthNodes.join(', ')} • ${injectedTotal.toFixed(1)} L/min injected`
                        : 'no leak injected'}
                    </dd>
                    <dt>Detected</dt>
                    <dd>
                      {detectedNodes.length
                        ? `${detectedNodes.length} site${detectedNodes.length > 1 ? 's' : ''}: ${detectedNodes.join(', ')}`
                        : 'none'}
                    </dd>
                    <dt>Total loss</dt>
                    <dd>
                      {(detection.total_loss_lpm ?? detection.leak_rate_lpm).toFixed(1)} L/min
                    </dd>
                  </dl>

                  {detectedNodes.length > 0 && (
                    <table className="data-table tb-table">
                      <thead>
                        <tr><th>Node</th><th>Rate</th><th>Severity</th><th>Conf.</th></tr>
                      </thead>
                      <tbody>
                        {(detection.leaks || []).map(l => (
                          <tr key={l.node}>
                            <td>
                              <strong>{l.node}</strong>
                              {leaks[l.node] ? null : <span className="tb-tag tb-tag-alt">not injected</span>}
                            </td>
                            <td>{l.rate_lpm.toFixed(1)}</td>
                            <td>{l.severity}</td>
                            <td>{(l.confidence * 100).toFixed(0)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  <dl className="tb-dl">
                  </dl>
                </>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">Live sensor readings</div>
            <table className="data-table tb-table">
              <thead>
                <tr>
                  <th>Junction</th><th>Flow</th><th>Pressure</th>
                  <th title="measured minus expected from mass balance">Residual</th>
                </tr>
              </thead>
              <tbody>
                {JUNCTIONS.map(jid => {
                  const residual = detection?.residuals?.[jid] ?? 0;
                  const threshold = detection?.thresholds?.[jid] ?? Infinity;
                  const flagged = residual > threshold;
                  return (
                    <tr key={jid} className={flagged ? 'tb-row-flagged' : ''}>
                      <td>
                        <strong>{jid}</strong>
                        {leaks[jid] ? <span className="tb-tag">leaking</span> : null}
                        {overrides[`flow_${jid}`] || overrides[`pressure_${jid}`]
                          ? <span className="tb-tag tb-tag-alt">forced</span> : null}
                      </td>
                      <td>{(flows[jid] ?? 0).toFixed(1)}</td>
                      <td>{(pressures[jid] ?? 0).toFixed(2)}</td>
                      <td className={flagged ? 'tb-residual-hot' : ''}>
                        {residual.toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

/** One machine row, with local edits committed on Apply. */
function MachineRow({ mid, machine, onApply }) {
  const [pct, setPct] = useState(null);
  const [st, setSt] = useState(null);

  // Follow the server until the user starts editing this row.
  const livePct = machine?.production_pct ?? 0;
  const liveSt = machine?.state ?? 'OFF';
  const shownPct = pct ?? livePct;
  const shownSt = st ?? liveSt;
  const dirty = shownPct !== livePct || shownSt !== liveSt;

  return (
    <tr>
      <td><strong>{mid}</strong></td>
      <td>
        <input className="tb-num" type="number" min="0" max="200" value={shownPct}
          onChange={e => setPct(Number(e.target.value))} />
      </td>
      <td>
        <select value={shownSt} onChange={e => setSt(e.target.value)}>
          {MACHINE_STATES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </td>
      <td>
        {(machine?.flow_lpm ?? 0).toFixed(0)}
        {dirty && (
          <button className="tb-btn tb-btn-sm tb-btn-apply"
            onClick={() => { onApply(shownPct, shownSt); setPct(null); setSt(null); }}>
            Apply
          </button>
        )}
      </td>
    </tr>
  );
}
