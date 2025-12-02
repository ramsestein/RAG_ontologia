import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, BookOpen, ChevronDown } from 'lucide-react';
import { APP_NAME } from '../../config/constants';

export const Navbar: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => 
    location.pathname === path 
      ? 'bg-white text-primary-600 shadow-sm ring-1 ring-slate-200' 
      : 'text-slate-500 hover:text-slate-900 hover:bg-slate-200/50';

  return (
    <nav className="bg-slate-100 border-b border-slate-200 h-16 flex items-center justify-between px-4 md:px-6 sticky top-0 z-50">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="bg-white p-1.5 rounded-lg shadow-sm ring-1 ring-slate-200">
          <Activity className="h-5 w-5 text-primary-600" />
        </div>
        <span className="font-bold text-lg text-slate-800 tracking-tight leading-none">
          {APP_NAME}
        </span>
      </div>

      {/* Navigation */}
      <div className="flex items-center gap-1 bg-slate-200/50 p-1 rounded-lg border border-slate-200/50">
        <Link 
          to="/" 
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive('/')}`}
        >
          <Activity className="h-4 w-4" />
          Workbench
        </Link>
        <Link 
          to="/docs" 
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive('/docs')}`}
        >
          <BookOpen className="h-4 w-4" />
          Documentation
        </Link>
      </div>

      {/* User Actions - Simplified */}
      <div className="flex items-center gap-3 pl-4 border-l border-slate-200">
        <div className="flex items-center gap-3 cursor-pointer hover:bg-slate-200/50 py-1 px-2 rounded-md transition-colors group">
          <div className="text-right hidden md:block">
            <p className="text-sm font-bold text-slate-700 group-hover:text-primary-600 transition-colors">Dr. Sarah L.</p>
            <p className="text-xs text-slate-500">Chief Resident</p>
          </div>
          <div className="relative">
             <div className="h-9 w-9 rounded-full bg-slate-200 border border-slate-300 flex items-center justify-center text-xs font-bold text-slate-600 overflow-hidden shadow-sm">
               <img src="https://i.pravatar.cc/150?u=a042581f4e29026704d" alt="Avatar" className="h-full w-full object-cover" />
             </div>
             <div className="absolute bottom-0 right-0 h-2.5 w-2.5 bg-green-500 rounded-full border-2 border-white"></div>
          </div>
          <ChevronDown className="h-4 w-4 text-slate-400 group-hover:text-slate-600 transition-colors" />
        </div>
      </div>
    </nav>
  );
};