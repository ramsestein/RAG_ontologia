import React from 'react';
import type { TocItem } from '../../config/docsConfig';

interface DocTableOfContentsProps {
  items: TocItem[];
  activeItemId?: string;
  onItemClick?: (itemId: string) => void;
}

/**
 * DocTableOfContents component following Single Responsibility Principle.
 * Renders the table of contents for the current documentation page.
 * 
 * Responsive: Hidden on mobile/tablet, visible on xl screens and above.
 */
export const DocTableOfContents: React.FC<DocTableOfContentsProps> = ({
  items,
  activeItemId,
  onItemClick,
}) => {
  const renderItem = (item: TocItem, depth: number = 0) => {
    const paddingClass = depth === 0 ? 'pl-4' : 'pl-8';
    const textClass =
      activeItemId === item.id
        ? 'text-primary-600 font-medium'
        : depth === 0
        ? 'text-slate-700 hover:text-primary-600 font-medium'
        : 'text-slate-400 hover:text-primary-600';

    return (
      <li key={item.id}>
        <button
          onClick={() => onItemClick?.(item.id)}
          className={`block ${paddingClass} ${textClass} text-left transition-colors py-1`}
        >
          {item.label}
        </button>
        {item.children && item.children.length > 0 && (
          <ul className="space-y-2 mt-2">
            {item.children.map((child) => renderItem(child, depth + 1))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <aside className="hidden xl:block w-64 shrink-0 h-[calc(100vh-4rem)] sticky top-16 pt-8 pr-8">
      <h5 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-primary-500" />
        On this page
      </h5>
      <ul className="space-y-3 text-sm border-l border-slate-200 ml-1">
        {items.map((item) => renderItem(item))}
      </ul>
    </aside>
  );
};
