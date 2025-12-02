import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  DocSidebar,
  DocMobileHeader,
  DocContent,
  DocTableOfContents,
} from '../../components/docs';
import { NAV_SECTIONS, DOC_PAGES } from '../../config/docsConfig';
import type { TocItem } from '../../config/docsConfig';

/**
 * DocLayout component following SOLID principles:
 * 
 * - Single Responsibility: Orchestrates doc page layout, delegates rendering to child components
 * - Open/Closed: New sections/content can be added via configuration
 * - Liskov Substitution: Child components are interchangeable
 * - Interface Segregation: Each component receives only the props it needs
 * - Dependency Inversion: Depends on abstractions (props interfaces), not concrete implementations
 */
export const DocLayout: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [activeNavItem, setActiveNavItem] = useState('installation');
  const [activeTocItem, setActiveTocItem] = useState('');
  const contentRef = useRef<HTMLDivElement>(null);

  // Get current page content
  const currentPage = DOC_PAGES[activeNavItem] || DOC_PAGES['installation'];

  // Get breadcrumbs based on active item
  const getBreadcrumbs = useCallback(() => {
    for (const section of NAV_SECTIONS) {
      const item = section.items.find(i => i.id === activeNavItem);
      if (item) {
        return ['Docs', section.title, item.label];
      }
    }
    return ['Docs'];
  }, [activeNavItem]);

  const handleToggleMobileMenu = useCallback(() => {
    setIsMobileMenuOpen((prev) => !prev);
  }, []);

  const handleCloseMobileMenu = useCallback(() => {
    setIsMobileMenuOpen(false);
  }, []);

  const handleNavItemClick = useCallback((itemId: string) => {
    setActiveNavItem(itemId);
    setActiveTocItem('');
    setIsMobileMenuOpen(false);
    // Scroll to top of content
    contentRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const handleTocItemClick = useCallback((itemId: string) => {
    setActiveTocItem(itemId);
    
    // Find the heading in the content and scroll to it
    const headingMap: Record<string, string> = {
      'overview': 'Overview',
      'system-requirements': 'System Requirements',
      'quick-start': 'Quick Start',
      'clone-repo': 'Clone the Repository',
      'install-deps': 'Install Dependencies',
      'start-app': 'Start the Application',
      'next-steps': 'Next Steps',
      'environment-variables': 'Environment Variables',
      'configuration-options': 'Configuration Options',
      'advanced-configuration': 'Advanced Configuration',
      'api-key-authentication': 'API Key Authentication',
      'oauth-2': 'OAuth 2.0',
      'role-based-access': 'Role-Based Access Control',
      'security-best-practices': 'Security Best Practices',
      'how-it-works': 'How It Works',
      'supported-entity-types': 'Supported Entity Types',
      'api-usage': 'API Usage',
      'confidence-scores': 'Confidence Scores',
      'architecture': 'Architecture',
      'pipeline-stages': 'Pipeline Stages',
      'text-preprocessing': 'Text Preprocessing',
      'entity-recognition': 'Entity Recognition',
      'vector-search': 'Vector Search',
      'performance-metrics': 'Performance Metrics',
      'optimization-tips': 'Optimization Tips',
      'how-vector-search-works': 'How Vector Search Works',
      'index-statistics': 'Index Statistics',
      'query-examples': 'Query Examples',
      'tuning-parameters': 'Tuning Parameters',
      'key-components': 'Key Components',
      'vector-store': 'The Vector Store',
      'inference-engine': 'The Inference Engine',
      'api-integration': 'API Integration',
      'data-privacy': 'Data Privacy',
    };

    const headingText = headingMap[itemId];
    if (headingText && contentRef.current) {
      const headings = contentRef.current.querySelectorAll('h1, h2, h3, h4');
      for (const heading of headings) {
        if (heading.textContent?.includes(headingText)) {
          heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
          break;
        }
      }
    }
  }, []);

  // Reset TOC item when page changes
  useEffect(() => {
    setActiveTocItem('');
  }, [activeNavItem]);

  return (
    <div className="flex flex-col lg:flex-row w-full min-h-screen bg-canvas">
      {/* Mobile Header */}
      <DocMobileHeader
        isMenuOpen={isMobileMenuOpen}
        onToggleMenu={handleToggleMobileMenu}
      />

      {/* Sidebar Navigation */}
      <DocSidebar
        isOpen={isMobileMenuOpen}
        onClose={handleCloseMobileMenu}
        sections={NAV_SECTIONS}
        activeItemId={activeNavItem}
        onItemClick={handleNavItemClick}
      />

      {/* Main Content */}
      <DocContent
        ref={contentRef}
        content={currentPage.content}
        breadcrumbs={getBreadcrumbs()}
      />

      {/* Table of Contents (Desktop) */}
      <DocTableOfContents
        items={currentPage.tocItems}
        activeItemId={activeTocItem}
        onItemClick={handleTocItemClick}
      />
    </div>
  );
};