import { api } from '../api/client';
import { usePolling } from './usePolling';

const INTERVAL = parseInt(import.meta.env.VITE_POLL_INTERVAL_MS || '2000', 10);

/**
 * Polls GET /state/current at the configured tick interval.
 * Returns { state, connected, error }.
 */
export function useNetworkState() {
  const { data, connected, error } = usePolling(
    () => api.getCurrentState(),
    INTERVAL
  );
  return { state: data, connected, error };
}
