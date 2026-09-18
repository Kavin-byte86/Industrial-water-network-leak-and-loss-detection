/**
 * Production vs consumption — the view that answers "is this a leak, or are we
 * just busy?".
 *
 * A bar per source: what production says the plant should draw, what it
 * actually drew, and the gap between them. When the plant ramps up, both bars
 * grow together and the gap stays flat — which is the whole point.
 */

const fmt = (n, d = 0) =>
  n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

export function ProductionPanel({ state }) {
  const production = state?.production;
  const detection = state?.detection;
  if (!production) return null;

  const measured = production.measured_total_lpm;
  const expected = production.expected_total_lpm;
  const unexplained = production.unexplained_lpm;
  const scale = Math.max(measured, expected, 1);

  const leaking = detection?.leak_detected;
  const abnormal = detection?.abnormal_consumption || [];

  return (
    <div className="card">
      <div className="card-header">Production vs consumption</div>

      <p className="prod-explainer">
        High flow is not a leak. Expected demand is computed from every machine's
        production rate and state, so a busy plant raises <em>both</em> bars.
        Only the gap can be lost water.
      </p>

      <div className="prod-bars">
        <div className="prod-row">
          <span className="prod-label">Expected from production</span>
          <div className="prod-track">
            <div className="prod-fill prod-fill-expected"
              style={{ width: `${(expected / scale) * 100}%` }} />
          </div>
          <span className="prod-value">{fmt(expected)} L/min</span>
        </div>

        <div className="prod-row">
          <span className="prod-label">Actually measured</span>
          <div className="prod-track">
            <div className="prod-fill prod-fill-measured"
              style={{ width: `${(measured / scale) * 100}%` }} />
          </div>
          <span className="prod-value">{fmt(measured)} L/min</span>
        </div>

        <div className="prod-row">
          <span className="prod-label">Unexplained</span>
          <div className="prod-track">
            <div className={`prod-fill ${leaking ? 'prod-fill-loss' : 'prod-fill-ok'}`}
              style={{ width: `${(Math.max(0, unexplained) / scale) * 100}%` }} />
          </div>
          <span className={`prod-value ${leaking ? 'prod-value-loss' : ''}`}>
            {fmt(unexplained, 1)} L/min
          </span>
        </div>
      </div>

      <div className="prod-facts">
        <span><label>Production load</label><strong>{production.production_load_pct}%</strong></span>
        <span><label>Machines running</label><strong>{production.machines_running} / 8</strong></span>
        <span><label>Demand accounted for</label><strong>{production.explained_pct}%</strong></span>
      </div>

      <p className={`prod-verdict ${leaking ? 'prod-verdict-loss' : 'prod-verdict-ok'}`}>
        {leaking
          ? `${fmt(Math.max(0, unexplained), 1)} L/min cannot be explained by current production — treated as loss.`
          : `All measured flow is accounted for by current production. No loss.`}
      </p>

      {abnormal.length > 0 && (
        <div className="prod-abnormal">
          <div className="prod-abnormal-title">
            Abnormal consumption vs historical pattern
          </div>
          <p className="tb-hint">
            These junctions balance correctly — no water is going missing — but
            they are drawing well above their own recent usage. Typically waste
            or a process change rather than a leak.
          </p>
          <table className="data-table">
            <thead>
              <tr><th>Junction</th><th>Now</th><th>Baseline</th><th>Deviation</th><th>σ</th></tr>
            </thead>
            <tbody>
              {abnormal.map(a => (
                <tr key={a.node}>
                  <td><strong>{a.node}</strong></td>
                  <td>{fmt(a.current_lpm, 1)}</td>
                  <td>{fmt(a.baseline_lpm, 1)}</td>
                  <td>+{fmt(a.deviation_lpm, 1)} L/min</td>
                  <td>{a.z_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
