import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Navbar } from '../components/layout/Navbar';
import { WorkbenchPage } from '../modules/workbench/WorkbenchPage';
import { DocLayout } from '../modules/docs/DocLayout';
import { DebugPage } from '../modules/debug/DebugPage';

export const AppRoutes: React.FC = () => {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<WorkbenchPage />} />
        <Route path="/docs" element={<DocLayout />} />
        <Route path="/debug" element={<DebugPage />} />
      </Routes>
    </>
  );
};