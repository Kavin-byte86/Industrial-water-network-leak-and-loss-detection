import { useState, useRef, useCallback, useEffect } from 'react';
import { useNetworkState } from './hooks/useNetworkState';
import { Sidebar } from './components/Sidebar/Sidebar';
import { PipelineSkeleton } from './components/PipelineSkeleton/PipelineSkeleton';
import { FacilityMap } from './components/FacilityMap/FacilityMap';
import { LiveFlowPanel } from './components/LiveFlowPanel/LiveFlowPanel';
import { HistoryView } from './components/HistoryView/HistoryView';

const VIEWS = ['pipeline', 'facility', 'liveflow', 'history'];

export default function App() {
  const [activeView, setActiveView] = useState('pipeline');
  const { state, connected, error } = useNetworkState();

  // Rolling buffer for sparklines (last 60 ticks)
  const bufferRef = useRef([]);
  useEffect(() => {
    if (state && state.flows) {
      bufferRef.current = [...bufferRef.current.slice(-59), state];
    }
  }, [state]);

  const renderView = () => {
    switch (activeView) {
      case 'pipeline':
        return <PipelineSkeleton state={state} />;
      case 'facility':
        return <FacilityMap state={state} />;
      case 'liveflow':
        return <LiveFlowPanel state={state} buffer={bufferRef.current} />;
      case 'history':
        return <HistoryView />;
      default:
        return null;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar
        activeView={activeView}
        onNavigate={setActiveView}
        state={state}
        connected={connected}
      />
      <main className="main-content">
        {renderView()}
      </main>
    </div>
  );
}
