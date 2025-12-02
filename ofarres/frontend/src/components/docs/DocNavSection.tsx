import React from 'react';

export interface DocNavItem {
  id: string;
  label: string;
}

interface DocNavSectionProps {
  title: string;
  items: DocNavItem[];
  activeItemId?: string;
  onItemClick?: (itemId: string) => void;
}

/**
 * DocNavSection component following Single Responsibility Principle.
 * Renders a single navigation section with its items.
 * 
 * Liskov Substitution: Can be replaced with any component that accepts the same props.
 */
export const DocNavSection: React.FC<DocNavSectionProps> = ({
  title,
  items,
  activeItemId,
  onItemClick,
}) => {
  return (
    <div>
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">
        {title}
      </h3>
      <ul className="space-y-1">
        {items.map((item) => (
          <li key={item.id}>
            <DocNavLink
              label={item.label}
              isActive={activeItemId === item.id}
              onClick={() => onItemClick?.(item.id)}
            />
          </li>
        ))}
      </ul>
    </div>
  );
};

interface DocNavLinkProps {
  label: string;
  isActive?: boolean;
  onClick?: () => void;
}

/**
 * DocNavLink component - Single Responsibility for rendering individual nav items.
 */
const DocNavLink: React.FC<DocNavLinkProps> = ({
  label,
  isActive = false,
  onClick,
}) => {
  return (
    <button
      onClick={onClick}
      className={`
        block w-full text-left pl-4 py-2 text-sm border-l-2
        transition-all duration-200
        ${
          isActive
            ? 'text-primary-700 border-primary-600 font-bold bg-primary-50 rounded-r-md'
            : 'text-slate-500 border-transparent hover:text-slate-900 hover:border-slate-300'
        }
      `}
    >
      {label}
    </button>
  );
};
