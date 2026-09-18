/**
 * Loss summary — the plant-level answer to "how much is this costing us".
 *
 * Sits above every view as a metrics strip. Rate answers "how bad right now",
 * volume answers "how much has gone", cost answers "why it matters".
 * All figures come from the backend's loss tracker; nothing is derived here
 * except formatting.
 */

const fmt = (n, digits = 0) =>
  n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

function Tile({ label, value, unit, tone, hint }) {
  return (
    <div className={`kpi ${tone ? `kpi-${tone}` : ''}`} title={hint}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">
        {value}
        {unit && <span className="kpi-unit">{unit}</span>}
      </div>
    </div>
  );
}

export function LossSummary({ losses, production }) {
  if (!losses || !production) return null;

  const sym = losses.currency_symbol || '';
  const leaking = losses.active_leaks > 0;

  return (
    <div className="kpi-row">
      <Tile
        label="Main inlet flow"
        value={fmt(production.measured_total_lpm)}
        unit="L/min"
        hint="Measured water entering the plant"
      />
      <Tile
        label="Expected from production"
        value={fmt(production.expected_total_lpm)}
        unit="L/min"
        hint={`${production.machines_running} machines running at ${production.production_load_pct}% average load`}
      />
      <Tile
        label="Unexplained"
        value={fmt(production.unexplained_lpm, 1)}
        unit="L/min"
        tone={leaking ? 'alarm' : 'good'}
        hint="Measured minus what the production schedule accounts for — the only part that can be a loss"
      />
      <Tile
        label="Water lost"
        value={fmt(losses.volume_lost_m3, 2)}
        unit="m³"
        tone={leaking ? 'alarm' : undefined}
        hint={`${fmt(losses.volume_lost_litres)} litres, including leaks already resolved`}
      />
      <Tile
        label="Cost so far"
        value={`${sym}${fmt(losses.cost_so_far)}`}
        tone={leaking ? 'alarm' : undefined}
        hint={`At ${sym}${losses.tariff_per_m3}/m³ (supply + effluent)`}
      />
      <Tile
        label={leaking ? 'Projected / day' : 'Projected / day'}
        value={`${sym}${fmt(losses.projected_daily_cost)}`}
        tone={leaking ? 'alarm' : 'good'}
        hint="If the current loss rate continues unrepaired for 24 hours"
      />
    </div>
  );
}
