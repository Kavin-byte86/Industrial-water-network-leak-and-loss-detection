/**
 * The trained classifier's verdict.
 *
 * This is the project's actual model — the 4-stage pipeline (Ridge expected-flow
 * regressors -> XGBoost leak classifier -> XGBoost zone locator -> XGBoost rate
 * regressor) running on live telemetry through the same feature code that built
 * its training set.
 *
 * The mass-balance detector runs alongside it as an independent physics check,
 * so this panel also reports whether the two agree. Agreement is evidence; a
 * disagreement is worth showing rather than hiding.
 */

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export function MLVerdict({ ml, detection }) {
  if (!ml) return null;

  if (!ml.available) {
    const label = {
      warming_up: 'Warming up',
      models_not_loaded: 'Model not loaded',
      prediction_error: 'Prediction failed',
    }[ml.reason] || 'Unavailable';

    return (
      <div className="card ml-card ml-card-idle">
        <div className="card-header">
          ML classifier
          <span className="ml-badge ml-badge-idle">{label}</span>
        </div>
        <p className="ml-detail">{ml.detail || 'No prediction available.'}</p>
      </div>
    );
  }

  const fired = ml.leak_detected;
  // The physics detector's opinion, for comparison.
  const physicsNodes = detection?.leaks?.map(l => l.node) || [];
  const physicsFired = physicsNodes.length > 0;
  const mlNode = ml.leak_zone ? ml.leak_zone.replace('ZONE_', '') : null;

  let agreement = null;
  if (fired && physicsFired) {
    agreement = physicsNodes.includes(mlNode)
      ? { kind: 'ok', text: `Mass balance agrees — both place it at ${mlNode}.` }
      : { kind: 'warn', text: `Mass balance puts it at ${physicsNodes.join(', ')} instead of ${mlNode}.` };
  } else if (fired && !physicsFired) {
    agreement = { kind: 'warn', text: 'Mass balance sees nothing — the model is ahead of the physics check, or wrong.' };
  } else if (!fired && physicsFired) {
    agreement = { kind: 'warn', text: `Mass balance flags ${physicsNodes.join(', ')} — the model has not fired.` };
  } else {
    agreement = { kind: 'ok', text: 'Mass balance agrees — network clean.' };
  }

  return (
    <div className={`card ml-card ${fired ? 'ml-card-alarm' : 'ml-card-clear'}`}>
      <div className="card-header">
        ML classifier
        <span className={`ml-badge ${fired ? 'ml-badge-alarm' : 'ml-badge-clear'}`}>
          {fired ? 'Leak predicted' : 'No leak'}
        </span>
      </div>

      <div className="ml-prob">
        <div className="ml-prob-bar">
          <div className="ml-prob-fill" style={{ width: pct(ml.leak_probability) }} />
          <div className="ml-prob-threshold" style={{ left: pct(ml.threshold ?? 0.5) }} />
        </div>
        <div className="ml-prob-labels">
          <span>P(leak) <strong>{pct(ml.leak_probability)}</strong></span>
          <span className="tb-muted">decision threshold {pct(ml.threshold ?? 0.5)}</span>
        </div>
      </div>

      {fired && (
        <div className="ml-facts">
          <span><label>Predicted zone</label><strong>{ml.leak_zone}</strong></span>
          <span><label>Predicted rate</label><strong>{ml.leak_rate_lpm?.toFixed(1)} L/min</strong></span>
        </div>
      )}

      <p className={`ml-agreement ml-agreement-${agreement.kind}`}>{agreement.text}</p>

      <div className="ml-meta">
        Stage 1 expected-flow → Stage 2 detect → Stage 3 localise → Stage 4 rate ·
        model <strong>{ml.model_version}</strong> · {ml.features_used} features ·
        {' '}{ml.history_ticks} ticks of context
      </div>
    </div>
  );
}
