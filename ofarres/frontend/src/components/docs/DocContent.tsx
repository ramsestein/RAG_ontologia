import React, { forwardRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChevronRight } from 'lucide-react';

interface DocContentProps {
  content: string;
  breadcrumbs?: string[];
}

/**
 * DocContent component following Single Responsibility Principle.
 * Renders the main documentation content with markdown support.
 * 
 * Dependency Inversion: Depends on content abstraction (string), not concrete data source.
 * Uses forwardRef to allow parent components to scroll content.
 */
export const DocContent = forwardRef<HTMLDivElement, DocContentProps>(({
  content,
  breadcrumbs = ['Docs', 'Getting Started'],
}, ref) => {
  return (
    <main 
      ref={ref}
      className="flex-1 min-w-0 p-4 md:p-6 xl:p-8 overflow-y-auto h-[calc(100vh-4rem)]"
    >
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 sm:p-6 md:p-8 lg:p-10 xl:p-12 mx-auto max-w-4xl">
        {/* Mobile Breadcrumbs */}
        <div className="mb-6 pb-6 border-b border-slate-100 lg:hidden">
          <div className="flex items-center gap-2 text-xs text-slate-400 flex-wrap">
            {breadcrumbs.map((crumb, index) => (
              <React.Fragment key={`${crumb}-${index}`}>
                {index > 0 && <ChevronRight className="h-3 w-3 flex-shrink-0" />}
                <span>{crumb}</span>
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Markdown Content */}
        <article className="prose prose-slate prose-headings:font-bold prose-headings:text-slate-900 prose-p:text-slate-600 prose-a:text-primary-600 hover:prose-a:text-primary-500 max-w-none prose-sm sm:prose-base prose-headings:scroll-mt-20">
          <ReactMarkdown>{content}</ReactMarkdown>
        </article>
      </div>
    </main>
  );
});

DocContent.displayName = 'DocContent';
