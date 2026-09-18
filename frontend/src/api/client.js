/**
 * API client — single fetch wrapper, base URL from env.
 *
 * Deployment topologies:
 *   1. VITE_API_BASE_URL set  → call that backend directly (CORS allows all).
 *      Use this for split-origin deploys (e.g. Vercel frontend + Render backend).
 *   2. Dev, no env var        → '/api', proxied by the Vite dev server.
 *   3. Production, no env var → '' (same origin): FastAPI serves this bundle
 *      and the API from one port.  If that is not the intended topology, set
 *      VITE_API_BASE_URL in Vercel Environment Variables to the Render URL.
 */

function resolveBaseUrl() {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl) return envUrl.replace(/\/+$/, '');    // trim trailing slashes

  if (import.meta.env.DEV) return '/api';           // Vite proxy

  // Production same-origin: works when FastAPI serves the bundle.
  // Log a warning so a misconfigured split-origin deploy is obvious.
  console.warn(
    '[api/client] VITE_API_BASE_URL is not set. API calls will go to the ' +
    'same origin. If the backend lives on a different host (e.g. Render), ' +
    'set VITE_API_BASE_URL in your Vercel Environment Variables.'
  );
  return '';
}

const BASE_URL = resolveBaseUrl();

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
