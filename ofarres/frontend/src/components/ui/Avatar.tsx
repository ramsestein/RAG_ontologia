import React from 'react';

interface AvatarProps {
  initials: string;
  size?: 'sm' | 'md' | 'lg';
  showStatus?: boolean;
  statusColor?: string;
  className?: string;
}

const sizeClasses = {
  sm: 'h-7 w-7 text-xs',
  md: 'h-9 w-9 text-xs',
  lg: 'h-12 w-12 text-sm',
};

/**
 * Avatar component following Single Responsibility Principle.
 * Displays user initials with optional online status indicator.
 */
export const Avatar: React.FC<AvatarProps> = ({
  initials,
  size = 'md',
  showStatus = false,
  statusColor = 'bg-green-500',
  className = '',
}) => {
  return (
    <div className={`relative ${className}`}>
      <div
        className={`
          ${sizeClasses[size]}
          rounded-full bg-gradient-to-br from-primary-500 to-primary-600
          border border-primary-400
          flex items-center justify-center
          font-bold text-white
          shadow-sm
        `}
      >
        {initials}
      </div>
      {showStatus && (
        <div
          className={`
            absolute bottom-0 right-0
            h-2.5 w-2.5
            ${statusColor}
            rounded-full
            border-2 border-white
          `}
        />
      )}
    </div>
  );
};
