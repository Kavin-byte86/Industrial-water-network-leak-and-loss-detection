import { useState, useRef, useEffect, useCallback } from 'react';
import { useNetworkState } from './hooks/useNetworkState';
import { Sidebar } from './components/Sidebar/Sidebar';
import { PipelineSkeleton } from './components/PipelineSkeleton/PipelineSkeleton';
import { FacilityMap } from './components/FacilityMap/FacilityMap';
import { LiveFlowPanel } from './components/LiveFlowPanel/LiveFlowPanel';
import { HistoryView } from './components/HistoryView/HistoryView';
import { LeakAlert } from './components/LeakAlert/LeakAlert';
import { LossSummary } from './components/LossSummary/LossSummary';
import { ProductionPanel } from './components/ProductionPanel/ProductionPanel';
import { MLVerdict } from './components/MLVerdict/MLVerdict';

const BUFFER_SIZE = 60;

export default function App() {
  const [activeView, setActiveView] = useState('pipeline');
  const { state, connected, error } = useNetworkState();

  // Rolling buffer for sparklines.
  //
  // This is state rather than a ref: a ref mutated inside an effect does not
  // trigger a re-render, so the sparklines rendered one tick behind the table
  // beside them. Keyed on timestamp so React StrictMode's double-invoked
  // effects (and any repeated poll of an unchanged tick) cannot append the
  // same tick twice.
  const [buffer, setBuffer] = useState([]);

  useEffect(() => {
    if (!state?.flows || !state.timestamp) return;
    setBuffer(prev => {
      if (prev.length && prev[prev.length - 1].timestamp === state.timestamp) {
        return prev;
      }
      return [...prev.slice(-(BUFFER_SIZE - 1)), state];
    });
  }, [state]);

  // Surfaces failures from write actions (control / simulation buttons), which
  // previously rejected silently and left the user with no idea why nothing
  // happened.
  const [actionError, setActionError] = useState(null);
  const dismissError = useCallback(() => setActionError(null), []);

  const detection = state?.detection || null;
  // Every leaking node, not just the primary — concurrent leaks must all be
  // visible on the diagrams.
  const leakNodes = detection?.leaks?.length
    ? detection.leaks.map(l => l.node)
    : detection?.leak_detected ? [detection.leak_node] : [];

  const renderView = () => {
    switch (activeView) {
      case 'pipeline':
        return <PipelineSkeleton state={state} leakNodes={leakNodes} />;
      case 'facility':
        return <FacilityMap state={state} onError={setActionError} leakNodes={leakNodes} />;
      case 'liveflow':
        return <LiveFlowPanel state={state} buffer={buffer} />;
      case 'analysis':
        return (
          <>
            <MLVerdict ml={state?.ml} detection={detection} />
            <ProductionPanel state={state} />
          </>
        );
      case 'history':
        return <HistoryView />;
      default:
        return null;
    }
  };

  // `state` carries a `detail` field instead of telemetry when the simulation
  // has not produced a tick yet.
  const awaitingData = connected && state && !state.flows;

  return (
    <div className="app-layout">
      <Sidebar
        activeView={activeView}
        onNavigate={setActiveView}
        state={state}
        connected={connected}
        onError={setActionError}
      />
      <main className="main-content">
        {!connected && (
          <div className="app-banner app-banner-error">
            <strong>Backend unreachable.</strong>{' '}
            {error || 'Retrying automatically…'} Start it with{' '}
            <code>uvicorn app.main:app --port 8000</code> from the{' '}
            <code>backend/</code> directory.
          </div>
        )}
        {awaitingData && (
          <div className="app-banner app-banner-info">
            Connected, waiting for the first simulation tick…
          </div>
        )}
        {actionError && (
          <div className="app-banner app-banner-error">
            <strong>Action failed.</strong> {actionError}
            <button className="app-banner-close" onClick={dismissError}>
              Dismiss
            </button>
          </div>
        )}
        {state?.flows && (
          <LossSummary losses={state.losses} production={state.production} />
        )}
        {state?.flows && <LeakAlert detection={detection} ml={state.ml} />}
        {renderView()}
      </main>
    </div>
  );
}
