/**
 * View 1 — Pipeline Skeleton (P&ID-style SVG schematic)
 *
 * Topology: J1 → J2/J3/J4/J7 → leaf junctions → machines/taps
 * Circles = junctions, Squares = machines, Triangles = taps
 * Animated flow dashes proportional to flow rate.
 */

import { useMemo } from 'react';

// ── Layout positions (hand-tuned for the exact topology) ──
const NODES = {
  J1:  { x: 450, y: 50,  type: 'junction' },
  J2:  { x: 180, y: 150, type: 'junction' },
  J3:  { x: 370, y: 150, type: 'junction' },
  J4:  { x: 560, y: 150, type: 'junction' },
  J7:  { x: 740, y: 150, type: 'junction' },
  J5:  { x: 120, y: 260, type: 'junction', endpoint: 'M1' },
  J6:  { x: 240, y: 260, type: 'junction', endpoint: 'M2' },
  J8:  { x: 300, y: 260, type: 'junction', endpoint: 'M3' },
  J9:  { x: 370, y: 260, type: 'junction', endpoint: 'M4' },
  J10: { x: 440, y: 260, type: 'junction', endpoint: 'M5' },
  J11: { x: 520, y: 260, type: 'junction', endpoint: 'M6' },
  J12: { x: 600, y: 260, type: 'junction', endpoint: 'M7' },
  J13: { x: 660, y: 260, type: 'junction', endpoint: 'M8' },
  J14: { x: 740, y: 260, type: 'junction', endpoint: 'T1' },
  J15: { x: 800, y: 260, type: 'junction', endpoint: 'T2' },
  J16: { x: 860, y: 260, type: 'junction', endpoint: 'T3' },
};

const ENDPOINTS = {
  M1: { x: 120, y: 350, type: 'machine', from: 'J5' },
  M2: { x: 240, y: 350, type: 'machine', from: 'J6' },
  M3: { x: 300, y: 350, type: 'machine', from: 'J8' },
  M4: { x: 370, y: 350, type: 'machine', from: 'J9' },
  M5: { x: 440, y: 350, type: 'machine', from: 'J10' },
  M6: { x: 520, y: 350, type: 'machine', from: 'J11' },
  M7: { x: 600, y: 350, type: 'machine', from: 'J12' },
  M8: { x: 660, y: 350, type: 'machine', from: 'J13' },
  T1: { x: 740, y: 350, type: 'tap', from: 'J14' },
  T2: { x: 800, y: 350, type: 'tap', from: 'J15' },
  T3: { x: 860, y: 350, type: 'tap', from: 'J16' },
};

const EDGES = [
  ['J1', 'J2'], ['J1', 'J3'], ['J1', 'J4'], ['J1', 'J7'],
  ['J2', 'J5'], ['J2', 'J6'],
  ['J3', 'J8'], ['J3', 'J9'], ['J3', 'J10'],
  ['J4', 'J11'], ['J4', 'J12'],
  ['J7', 'J13'], ['J7', 'J14'], ['J7', 'J15'], ['J7', 'J16'],
];

const ENDPOINT_EDGES = Object.entries(ENDPOINTS).map(([id, ep]) => [ep.from, id]);

export function PipelineSkeleton({ state }) {
  const flows = state?.flows || {};
  const machines = state?.machines || {};
  const taps = state?.taps || {};

  return (
    <div>
      <div className="view-header">
        <h2 className="view-title">Pipeline Skeleton</h2>
      </div>
      <svg className="pipeline-svg" viewBox="0 0 960 420" preserveAspectRatio="xMidYMid meet">
        {/* Title */}
        <text x="480" y="30" textAnchor="middle" fontSize="11" fontWeight="600" fill="#4a5568">
          Water Network Topology — J1 Main Inlet
        </text>

        {/* Junction-to-junction edges */}
        {EDGES.map(([from, to]) => {
          const a = NODES[from], b = NODES[to];
          const flow = Math.abs(flows[to] || 0);
          const isActive = flow > 0.5;
          const speed = isActive ? Math.max(0.3, Math.min(2, 1 / (flow / 200))) : 0;
          return (
            <line key={`${from}-${to}`}
              x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              className={isActive ? 'pipe-flow' : 'pipe-idle'}
              style={isActive ? { animationDuration: `${speed}s` } : {}}
            />
          );
        })}

        {/* Junction-to-endpoint edges */}
        {ENDPOINT_EDGES.map(([from, to]) => {
          const a = NODES[from], b = ENDPOINTS[to];
          const jFlow = Math.abs(flows[from] || 0);
          const isActive = jFlow > 0.5;
          const speed = isActive ? Math.max(0.3, Math.min(2, 1 / (jFlow / 200))) : 0;
          return (
            <line key={`${from}-${to}`}
              x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              className={isActive ? 'pipe-flow' : 'pipe-idle'}
              style={isActive ? { animationDuration: `${speed}s` } : {}}
            />
          );
        })}

        {/* Junction nodes (circles) */}
        {Object.entries(NODES).map(([id, node]) => {
          const flow = flows[id] || 0;
          return (
            <g key={id}>
              <circle cx={node.x} cy={node.y} r={14} className="node-junction" />
              <text className="node-label" x={node.x} y={node.y + 3}>{id}</text>
              <text className="flow-label" x={node.x} y={node.y - 20}>
                {flow > 0 ? flow.toFixed(0) : ''}
              </text>
            </g>
          );
        })}

        {/* Machine nodes (squares) */}
        {Object.entries(ENDPOINTS).filter(([, ep]) => ep.type === 'machine').map(([id, ep]) => {
          const mState = machines[id]?.state || 'OFF';
          const isRunning = mState === 'RUNNING';
          return (
            <g key={id}>
              <rect
                x={ep.x - 14} y={ep.y - 14} width={28} height={28} rx={3}
                className={`node-machine ${isRunning ? 'running' : ''}`}
              />
              <text className="node-label" x={ep.x} y={ep.y + 3}>{id}</text>
              <text className="flow-label" x={ep.x} y={ep.y + 30}>
                {machines[id]?.flow_lpm?.toFixed(0) || '0'} L/m
              </text>
            </g>
          );
        })}

        {/* Tap nodes (triangles) */}
        {Object.entries(ENDPOINTS).filter(([, ep]) => ep.type === 'tap').map(([id, ep]) => {
          const tState = taps[id]?.state || 'CLOSED';
          const isOpen = tState === 'OPEN';
          const points = `${ep.x},${ep.y - 14} ${ep.x - 14},${ep.y + 12} ${ep.x + 14},${ep.y + 12}`;
          return (
            <g key={id}>
              <polygon
                points={points}
                className={`node-tap ${isOpen ? 'open' : ''}`}
              />
              <text className="node-label" x={ep.x} y={ep.y + 5}>{id}</text>
              <text className="flow-label" x={ep.x} y={ep.y + 30}>
                {taps[id]?.flow_lpm?.toFixed(0) || '0'} L/m
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform="translate(20, 390)">
          <circle cx={0} cy={0} r={6} className="node-junction" />
          <text x={12} y={4} fontSize="8" fill="#718096">Junction</text>
          <rect x={60} y={-6} width={12} height={12} rx={2} className="node-machine" />
          <text x={78} y={4} fontSize="8" fill="#718096">Machine</text>
          <polygon points="130,-6 124,8 136,8" className="node-tap" />
          <text x={142} y={4} fontSize="8" fill="#718096">Tap</text>
        </g>
      </svg>
    </div>
  );
}
