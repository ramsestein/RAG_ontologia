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
      // Installation
      'overview': 'Overview',
      'system-requirements': 'System Requirements',
      'quick-start': 'Quick Start',
      'access-points': 'Access Points',
      
      // Architecture Overview
      'two-stage-philosophy': 'The Two-Stage Philosophy',
      'why-two-stages': 'Why Two Stages?',
      'pipeline-architecture': 'Pipeline Architecture',
      'key-insight': 'Key Insight',
      
      // Why 100% Recall
      'catastrophic-cost': 'The Catastrophic Cost of Missing Entities',
      'low-precision-acceptable': 'Why Low Precision',
      'ner-performance': 'Our NER Performance',
      'golden-rule': 'The Golden Rule',
      
      // NER Workers
      'multi-worker-strategy': 'Multi-Worker Strategy',
      'ontology-ner': 'OntologyNER',
      'scispacy-ner': 'ScispaCyNER',
      'acronym-ner': 'AcronymNER',
      'assembly-effect': 'The Assembly Effect',
      'confidence-tiers': 'Confidence Tiers',
      
      // Error Stacking
      'errors-propagate': 'Errors Propagate and Amplify',
      'ner-errors-fatal': 'NER Errors Are Fatal',
      'rag-errors-recoverable': 'RAG Errors Are Recoverable',
      'asymmetry': 'The Asymmetry',
      'conclusion': 'Conclusion',
      
      // RAG Precision
      'stage-2-goals': 'Stage 2 Goals',
      'why-precision-matters': 'Why Precision Matters Here',
      'rag-filtering-process': 'RAG Filtering Process',
      'precision-in-action': 'Precision in Action',
      'why-rag-recall-less-critical': 'Why RAG Recall Is Less Critical',
      
      // Dictionary Power
      'why-dictionary-gold': 'Why Dictionary Matches Are Gold',
      'high-precision-foundation': 'The High-Precision Foundation',
      'experimental-evidence': 'Experimental Evidence',
      'the-takeaway': 'The Takeaway',
      'why-matters-rag': 'Why This Matters for RAG',
      'dictionary-statistics': 'Dictionary Statistics',
      
      // LLM Weighting
      'how-entities-weighted': 'How Entities Are Weighted',
      'why-weighting-works': 'Why This Weighting Works',
      'context-assembly': 'Context Assembly for LLM',
      'llm-intelligence': "The LLM's Intelligence",
      'example-comparison': 'Example Comparison',
      'practical-impact': 'Practical Impact',
      
      // Pipeline Example
      'input-clinical-note': 'Input Clinical Note',
      'stage-1-ner-processing': 'Stage 1: NER Processing',
      'stage-2-rag-processing': 'Stage 2: RAG Processing',
      'final-llm-output': 'Final LLM Output',
      'quality-improvement': 'Quality Improvement',
      
      // Performance Metrics
      'understanding-metrics': 'Understanding the Metrics',
      'current-performance': 'Our Current Performance',
      'why-numbers-make-sense': 'Why These Numbers Make Sense',
      'performance-summary': 'Performance Summary',
      'key-takeaways': 'Key Takeaways',
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