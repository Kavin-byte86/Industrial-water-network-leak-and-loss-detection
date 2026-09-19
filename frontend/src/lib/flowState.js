/**
 * When is a pipe actually carrying water?
 *
 * Machines draw a small standby trickle while switched OFF (0.1-0.4 L/min
 * each, from `classifier/config.py`, matching how the training data models an
 * idle machine). Those trickles add up the tree, so with the whole plant shut
 * down the meters still read:
 *
 *     J1 1.90   J3 0.90   J2 0.50   J4 0.30   J7 0.20   (L/min)
 *
 * The views used to call a pipe active at `flow > 0` or `flow > 0.5`, both of
 * which sit inside that idle band. J2's 0.50 landed exactly on the 0.5 cut, so
 * 2% meter noise flipped it back and forth and the J1-J2 link flickered on for
 * a tick at a time with nothing running. J1 and J3 sat above the cut
 * permanently and animated continuously.
 *
 * Standby peaks at 1.90 L/min (~2.0 with noise) and the smallest genuine draw
 * is 5.00 L/min (a machine STOPPING), so the threshold below sits in the gap:
 * high enough that no combination of idle trickle and noise can cross it, low
 * enough that every real flow does.
 */

/** Idle standby cannot exceed this; any real draw comfortably clears it. */
export const ACTIVE_FLOW_LPM = 3.0;

/** True when a junction is carrying real water rather than standby trickle. */
export function isFlowing(flow) {
  return Number.isFinite(flow) && flow > ACTIVE_FLOW_LPM;
}

/** `pipe-flow` when water is moving, `pipe-idle` otherwise. */
export function pipeClass(flow) {
  return isFlowing(flow) ? 'pipe-flow' : 'pipe-idle';
}

/**
 * Dash animation period: faster for heavier flow, clamped so it never becomes
 * a strobe or a crawl. Returns null when idle, so callers apply no animation.
 */
export function flowAnimation(flow, { min = 0.3, max = 2 } = {}) {
  if (!isFlowing(flow)) return null;
  return `${Math.max(min, Math.min(max, 200 / flow))}s`;
}
