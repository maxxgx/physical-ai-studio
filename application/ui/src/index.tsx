import React from 'react';

import ReactDOM from 'react-dom/client';

import './dev/runtime-sessions-fetch-mock';

import { Providers } from './providers';

import './index.css';

const rootEl = document.getElementById('root');
if (rootEl) {
    const root = ReactDOM.createRoot(rootEl);
    root.render(
        <React.StrictMode>
            <Providers />
        </React.StrictMode>
    );
}
