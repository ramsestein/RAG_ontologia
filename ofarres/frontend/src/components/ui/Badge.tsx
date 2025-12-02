import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  colorClass?: string;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ 
  children, 
  colorClass = 'bg-slate-100 text-slate-800',
  className = ''
}) => {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border border-transparent ${colorClass} ${className}`}>
      {children}
    </span>
  );
};