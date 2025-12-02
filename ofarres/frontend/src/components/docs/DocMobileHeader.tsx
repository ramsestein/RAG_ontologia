import React from 'react';
import { Book, Menu, X } from 'lucide-react';

interface DocMobileHeaderProps {
  isMenuOpen: boolean;
  onToggleMenu: () => void;
}

/**
 * DocMobileHeader component following Single Responsibility Principle.
 * Handles mobile-only header with menu toggle for documentation pages.
 * 
 * Interface Segregation: Only exposes what's needed for menu control.
 */
export const DocMobileHeader: React.FC<DocMobileHeaderProps> = ({
  isMenuOpen,
  onToggleMenu,
}) => {
  return (
    <div className="lg:hidden bg-white border-b border-slate-200 p-4 sticky top-16 z-30 flex items-center justify-between shadow-sm">
      <span className="font-bold text-slate-700 flex items-center gap-2">
        <Book className="h-4 w-4 text-primary-600" />
        Documentation
      </span>
      <button
        onClick={onToggleMenu}
        className="p-2 text-slate-500 hover:bg-slate-100 rounded-md transition-colors"
        aria-label={isMenuOpen ? 'Close menu' : 'Open menu'}
        aria-expanded={isMenuOpen}
      >
        {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>
    </div>
  );
};
