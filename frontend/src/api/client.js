/**
 * API client — single fetch wrapper, base URL from env.
 */

// Three cases, in priority order:
//   1. VITE_BACKEND_URL set  → call that backend directly (its CORS allows all).
//   2. dev, no env var       → '/api', which the Vite dev server proxies.
//   3. production build      → '' (same origin): FastAPI serves this bundle
//                              itself, so '/state/current' hits the API directly.
// .env.production blanks VITE_BACKEND_URL so a dev .env cannot bake a
// localhost URL into the production bundle.
const BASE_URL =
  import.meta.env.VITE_BACKEND_URL || (import.meta.env.DEV ? '/api' : '');

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export const api = {
  // Network
  getTopology: () => request('/network/topology'),

  // State
  getCurrentState: () => request('/state/current'),
  getHistory: (limit = 100) => request(`/state/history?limit=${limit}`),

  // Control
  controlMachine: (machine_id, production_pct, state) =>
    request('/control/machine', {
      method: 'POST',
      body: JSON.stringify({ machine_id, production_pct, state }),
    }),
  controlTap: (tap_id, state) =>
    request('/control/tap', {
      method: 'POST',
      body: JSON.stringify({ tap_id, state }),
    }),

  // Simulation
  getSimStatus: () => request('/simulation/status'),
  step: () => request('/simulation/step', { method: 'POST' }),
  pause: () => request('/simulation/pause', { method: 'POST' }),
  resume: () => request('/simulation/resume', { method: 'POST' }),
  reset: () => request('/simulation/reset', { method: 'POST' }),

  // Datasink
  getDatasinkLatest: () => request('/datasink/latest'),
  getDatasinkExport: (since_ticks = 100) =>
    request(`/datasink/export?since_ticks=${since_ticks}`),

  // Predict
  getPrediction: () => request('/predict/current'),

  // Test bench — leak injection, sensor overrides, endpoint control
  getTestbenchSnapshot: () => request('/testbench/snapshot'),
  setLeak: (node_id, rate_lpm) =>
    request('/testbench/leak', {
      method: 'POST',
      body: JSON.stringify({ node_id, rate_lpm }),
    }),
  clearLeak: (node_id) =>
    request(`/testbench/leak/${node_id}`, { method: 'DELETE' }),
  clearAllLeaks: () => request('/testbench/leaks/clear', { method: 'POST' }),
  setOverride: (sensor, value) =>
    request('/testbench/override', {
      method: 'POST',
      body: JSON.stringify({ sensor, value }),
    }),
  clearOverride: (sensor) =>
    request(`/testbench/override/${sensor}`, { method: 'DELETE' }),
  clearAllOverrides: () =>
    request('/testbench/overrides/clear', { method: 'POST' }),
  setEndpoint: (endpoint_id, production_pct, state) =>
    request('/testbench/endpoint', {
      method: 'POST',
      body: JSON.stringify({ endpoint_id, production_pct, state }),
    }),
};
