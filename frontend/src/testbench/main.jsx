import React from 'react';
import ReactDOM from 'react-dom/client';
import TestBench from './TestBench';
import '../styles/theme.css';
import './testbench.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <TestBench />
  </React.StrictMode>
);
