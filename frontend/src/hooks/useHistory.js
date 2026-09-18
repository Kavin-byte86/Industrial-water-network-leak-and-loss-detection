import { api } from '../api/client';
import { usePolling } from './usePolling';

/**
 * Polls GET /state/history at a slower interval (5s).
 * Returns { history, error }.
 */
export function useHistory(limit = 200) {
  const { data, error } = usePolling(
    () => api.getHistory(limit),
    5000
  );
  return {
    history: data?.ticks || [],
    count: data?.count || 0,
    error,
  };
}
