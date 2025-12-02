import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, BookOpen } from 'lucide-react';
import { APP_NAME } from '../../config/constants';

/**
 * Navbar component following Single Responsibility Principle.
 * Handles top-level navigation layout only.
 * 
 * Open/Closed Principle: New nav items can be added via configuration.
 */
export const Navbar: React.FC = () => {
  const location = useLocation();

  const getNavLinkClasses = (path: string): string => {
    const isActive = location.pathname === path;
    const baseClasses = 'px-3 sm:px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-1 sm:gap-2';
    const activeClasses = 'bg-white text-primary-600 shadow-sm ring-1 ring-slate-200';
    const inactiveClasses = 'text-slate-500 hover:text-slate-900 hover:bg-slate-200/50';
    
    return `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`;
  };

  return (
    <nav className="bg-slate-100 border-b border-slate-200 h-14 sm:h-16 flex items-center justify-center px-3 sm:px-4 md:px-6 sticky top-0 z-50">
      {/* Brand */}
      <div className="flex items-center gap-2 sm:gap-3 absolute left-3 sm:left-4 md:left-6">
        <div className="bg-white p-1 sm:p-1.5 rounded-lg shadow-sm ring-1 ring-slate-200">
          <Activity className="h-4 w-4 sm:h-5 sm:w-5 text-primary-600" />
        </div>
        <span className="font-bold text-base sm:text-lg text-slate-800 tracking-tight leading-none hidden sm:inline">
          {APP_NAME}
        </span>
      </div>

      {/* Navigation - Centered */}
      <div className="flex items-center gap-1 bg-slate-200/50 p-1 rounded-lg border border-slate-200/50">
        <Link to="/" className={getNavLinkClasses('/')}>
          <Activity className="h-4 w-4" />
          <span className="hidden sm:inline">Workbench</span>
        </Link>
        <Link to="/docs" className={getNavLinkClasses('/docs')}>
          <BookOpen className="h-4 w-4" />
          <span className="hidden sm:inline">Docs</span>
        </Link>
      </div>
    </nav>
  );
};