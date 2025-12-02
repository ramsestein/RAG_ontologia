import React from 'react';
import { ChevronDown } from 'lucide-react';
import { Avatar } from './Avatar';

interface UserMenuProps {
  name: string;
  role: string;
  initials: string;
  isOnline?: boolean;
}

/**
 * UserMenu component following Single Responsibility Principle.
 * Handles user display in navigation - separate from Avatar logic.
 */
export const UserMenu: React.FC<UserMenuProps> = ({
  name,
  role,
  initials,
  isOnline = true,
}) => {
  return (
    <div className="flex items-center gap-3 pl-4 border-l border-slate-200">
      <div className="flex items-center gap-3 cursor-pointer hover:bg-slate-200/50 py-1 px-2 rounded-md transition-colors group">
        <div className="text-right hidden md:block">
          <p className="text-sm font-bold text-slate-700 group-hover:text-primary-600 transition-colors">
            {name}
          </p>
          <p className="text-xs text-slate-500">{role}</p>
        </div>
        <Avatar
          initials={initials}
          size="md"
          showStatus={isOnline}
          statusColor="bg-green-500"
        />
        <ChevronDown className="h-4 w-4 text-slate-400 group-hover:text-slate-600 transition-colors" />
      </div>
    </div>
  );
};
