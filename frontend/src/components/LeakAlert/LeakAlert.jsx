/**
 * Leak alert — the operator-facing result of detection.
 *
 * Shown above every view so a leak cannot be missed by being on the wrong tab.
 * Values come from the backend's mass-balance detector (see
 * backend/app/simulation/detection.py); nothing is computed here.
 *
 * Concurrent leaks are listed individually. Residuals are independent per
 * junction, so two simultaneous leaks are two findings — showing only the
 * biggest would hide real water loss.
 */

const ZONE_DESCRIPTION = {
  J1: 'Main inlet header',
  J2: 'Branch A manifold (M1, M2)',
  J3: 'Branch B manifold (M3, M4, M5)',
  J4: 'Branch C manifold (M6, M7)',
  J7: 'Branch D utility manifold (M8, taps)',
  J5: 'M1 Scouring feed line',
  J6: 'M2 Bleaching feed line',
  J8: 'M3 Dyeing 1 feed line',
  J9: 'M4 Dyeing 2 feed line',
  J10: 'M5 Washing feed line',
  J11: 'M6 Finishing feed line',
  J12: 'M7 Washing/rinsing feed line',
  J13: 'M8 Utility feed line',
  J14: 'Tap 1 washdown line',
  J15: 'Tap 2 cleaning line',
  J16: 'Tap 3 sampling line',
};

const SEVERITY_RANK = { Small: 1, Medium: 2, Large: 3 };

function Fact({ label, value }) {
  return (
    <span>
      <label>{label}</label>
      <strong>{value}</strong>
    </span>
  );
}

export function LeakAlert({ detection, ml = null, currency = '₹' }) {
  if (!detection) return null;

  // Older payloads only carried the primary leak; fall back to it so the
  // dashboard still renders against a backend that predates multi-leak.
  const leaks = detection.leaks?.length
    ? detection.leaks
    : detection.leak_detected
      ? [{
          node: detection.leak_node,
          zone: detection.leak_zone,
          endpoint: detection.endpoint,
          rate_lpm: detection.leak_rate_lpm,
          severity: detection.severity,
          confidence: detection.confidence,
        }]
      : [];

  if (!leaks.length) {
    return (
      <div className="leak-alert leak-alert-clear">
        <span className="leak-alert-icon" aria-hidden="true">✓</span>
        <span>
          <strong>Network normal.</strong> Mass balance holds at all 16
          junctions, and the ML classifier reports no leak
          {ml?.available ? ` (P=${(ml.leak_probability * 100).toFixed(1)}%)` : ''}.
        </span>
      </div>
    );
  }

  // The banner takes the worst severity present, not the primary leak's.
  const worst = leaks.reduce(
    (acc, l) => (SEVERITY_RANK[l.severity] > SEVERITY_RANK[acc] ? l.severity : acc),
    'Small'
  );
  const total = detection.total_loss_lpm
    ?? leaks.reduce((a, l) => a + l.rate_lpm, 0);
  const multi = leaks.length > 1;

  return (
    <div className={`leak-alert leak-alert-${worst.toLowerCase()}`} role="alert">
      <span className="leak-alert-icon" aria-hidden="true">!</span>
      <div className="leak-alert-body">
        <div className="leak-alert-title">
          {multi
            ? `${leaks.length} simultaneous leaks detected — worst is ${worst} severity`
            : `Leak detected — ${worst} severity`}
          {ml?.available && (
            <span className={`leak-alert-source ${ml.leak_detected ? 'confirmed' : 'unconfirmed'}`}>
              {ml.leak_detected
                ? `ML model confirms · P=${(ml.leak_probability * 100).toFixed(0)}%`
                : `ML model has not fired · P=${(ml.leak_probability * 100).toFixed(0)}%`}
            </span>
          )}
        </div>

        {multi && (
          <div className="leak-alert-facts leak-alert-total">
            <Fact label="Total loss" value={`${total.toFixed(1)} L/min`} />
            <Fact label="Per hour" value={`${(total * 60).toFixed(0)} L/h`} />
            <Fact label="Sites" value={leaks.map(l => l.node).join(', ')} />
          </div>
        )}

        <ol className={multi ? 'leak-list' : 'leak-list leak-list-single'}>
          {leaks.map(leak => (
            <li key={leak.node} className={`leak-item leak-item-${(leak.severity || '').toLowerCase()}`}>
              {multi && (
                <span className="leak-item-rank">{leak.severity}</span>
              )}
              <div className="leak-alert-facts">
                <Fact label="Location" value={leak.node} />
                <Fact label="Zone" value={ZONE_DESCRIPTION[leak.node] || leak.zone} />
                {leak.endpoint && <Fact label="Serves" value={leak.endpoint} />}
                <Fact label="Loss rate" value={`${leak.rate_lpm.toFixed(1)} L/min`} />
                {leak.volume_lost_litres != null && (
                  <Fact label="Water lost"
                    value={`${leak.volume_lost_m3.toFixed(2)} m³`} />
                )}
                {leak.duration_hours != null && (
                  <Fact label="Running for" value={`${leak.duration_hours.toFixed(1)} h`} />
                )}
                {leak.cost_so_far != null && (
                  <Fact label="Cost so far" value={`${currency}${leak.cost_so_far.toLocaleString()}`} />
                )}
                {leak.projected_daily_cost != null && (
                  <Fact label="If unrepaired"
                    value={`${currency}${Math.round(leak.projected_daily_cost).toLocaleString()}/day`} />
                )}
                {leak.pressure?.corroborates && (
                  <Fact label="Pressure drop"
                    value={`${leak.pressure.drop_bar.toFixed(2)} bar`} />
                )}
                {!multi && (
                  <Fact label="Severity" value={leak.severity} />
                )}
                <Fact label="Confidence" value={`${(leak.confidence * 100).toFixed(0)}%`} />
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
