import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Generic polling hook — fetches from a function at a configurable interval.
 * @param {Function} fetchFn — async function that returns data
 * @param {number} intervalMs — poll interval in ms
 * @param {boolean} enabled — whether polling is active
 */
export function usePolling(fetchFn, intervalMs = 2000, enabled = true) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [connected, setConnected] = useState(false);
  const savedFn = useRef(fetchFn);

  useEffect(() => { savedFn.current = fetchFn; }, [fetchFn]);

  const poll = useCallback(async () => {
    try {
      const result = await savedFn.current();
      setData(result);
      setError(null);
      setConnected(true);
    } catch (err) {
      setError(err.message);
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    poll(); // immediate first fetch
    const id = setInterval(poll, intervalMs);
    return () => clearInterval(id);
  }, [poll, intervalMs, enabled]);

  return { data, error, connected, refetch: poll };
}
