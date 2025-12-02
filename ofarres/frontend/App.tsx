import React from 'react';
import { HashRouter } from 'react-router-dom';
import { AppRoutes } from './src/routes/AppRoutes';

const App: React.FC = () => {
  return (
    <HashRouter>
      <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
        <AppRoutes />
      </div>
    </HashRouter>
  );
};

export default App;