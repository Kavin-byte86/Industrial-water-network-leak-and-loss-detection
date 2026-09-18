/**
 * API client — single fetch wrapper, base URL from env.
 */

const BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

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
};
