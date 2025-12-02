import React, { useState, useMemo, useCallback } from 'react';
import { Search, Book, X } from 'lucide-react';
import { DocNavSection as DocNavSectionComponent } from './DocNavSection';
import type { DocNavItem } from './DocNavSection';

export interface DocNavSection {
  id: string;
  title: string;
  items: DocNavItem[];
}

export type { DocNavItem };

interface SearchResult {
  id: string;
  label: string;
  section: string;
}

interface DocSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  sections: DocNavSection[];
  activeItemId?: string;
  onItemClick?: (itemId: string) => void;
}

/**
 * DocSidebar component following Single Responsibility Principle.
 * Handles sidebar layout, search, and navigation.
 */
export const DocSidebar: React.FC<DocSidebarProps> = ({
  isOpen,
  onClose,
  sections,
  activeItemId,
  onItemClick,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  // Generate searchable items from sections
  const searchableItems = useMemo((): SearchResult[] => {
    const items: SearchResult[] = [];
    sections.forEach(section => {
      section.items.forEach(item => {
        items.push({
          id: item.id,
          label: item.label,
          section: section.title,
        });
      });
    });
    return items;
  }, [sections]);

  // Filter sections for display based on search
  const filteredSections = useMemo(() => {
    if (!searchQuery.trim()) return sections;
    
    const query = searchQuery.toLowerCase();
    return sections
      .map(section => ({
        ...section,
        items: section.items.filter(item => 
          item.label.toLowerCase().includes(query)
        ),
      }))
      .filter(section => section.items.length > 0);
  }, [sections, searchQuery]);

  // Check if we have search results
  const hasSearchResults = searchQuery.trim() && filteredSections.length > 0;
  const noSearchResults = searchQuery.trim() && filteredSections.length === 0;

  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  }, []);

  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
  }, []);

  const handleItemClick = useCallback((id: string) => {
    onItemClick?.(id);
    setSearchQuery('');
  }, [onItemClick]);

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/50 z-40 lg:hidden backdrop-blur-sm"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-50 w-72
          bg-white border-r border-slate-200
          transform transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:static lg:z-auto
          lg:h-[calc(100vh-4rem)] lg:sticky lg:top-16
          lg:bg-transparent lg:border-none lg:shadow-none
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        <div className="h-full lg:h-auto lg:m-4 xl:m-6 lg:bg-white lg:rounded-xl lg:shadow-sm lg:border lg:border-slate-200 overflow-hidden flex flex-col">
          {/* Header with Search */}
          <div className="p-4 lg:p-5 border-b border-slate-100 bg-slate-50/50">
            <h2 className="font-bold text-slate-900 flex items-center gap-2 mb-4">
              <Book className="h-5 w-5 text-primary-600" />
              <span>Doc Explorer</span>
            </h2>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={handleSearchChange}
                placeholder="Search docs..."
                className="w-full pl-9 pr-9 py-2 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
              />
              {searchQuery && (
                <button
                  onClick={handleClearSearch}
                  className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600 transition-colors"
                  aria-label="Clear search"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>

          {/* Navigation Sections */}
          <nav className="flex-1 overflow-y-auto p-4 lg:p-5 space-y-6 lg:space-y-8 bg-white custom-scrollbar">
            {noSearchResults ? (
              <div className="text-center py-8 text-slate-400">
                <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No results found</p>
                <p className="text-xs mt-1">Try a different search term</p>
              </div>
            ) : (
              filteredSections.map((section) => (
                <DocNavSectionComponent
                  key={section.id}
                  title={section.title}
                  items={section.items}
                  activeItemId={activeItemId}
                  onItemClick={handleItemClick}
                />
              ))
            )}
          </nav>
        </div>
      </aside>
    </>
  );
};
