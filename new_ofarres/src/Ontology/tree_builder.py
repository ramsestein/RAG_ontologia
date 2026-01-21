# -*- coding: utf-8 -*-
"""
Ontology Tree Builder for Spanish Radiology NER.

Uses owlready2 to load a SNOMED-based OWL ontology and constructs
hierarchical trees from extracted entities.
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

try:
    from owlready2 import get_ontology, Thing
except ImportError:
    print("[ERROR] owlready2 not installed. Install with: pip install owlready2")
    raise


class OntologyTree:
    """
    Builds hierarchical trees from NER entities using an OWL ontology.
    
    Maps extracted entity codes to ontology classes and constructs
    lineages (paths to root) for tree visualization.
    """
    
    def __init__(self, owl_path: str):
        """
        Initialize the ontology tree builder.
        
        Args:
            owl_path: Path to the OWL ontology file
        """
        print(f"[OntologyTree] Loading ontology from: {owl_path}")
        
        if not os.path.exists(owl_path):
            raise FileNotFoundError(f"OWL file not found: {owl_path}")
        
        # Load ontology
        self.onto = get_ontology(f"file://{owl_path}").load()
        print(f"[OK] Ontology loaded: {self.onto.base_iri}")
        
        # Build code-to-class lookup
        self._build_code_lookup()
    
    def _build_code_lookup(self):
        """Build a dictionary mapping codes to OWL classes."""
        self.code_to_class = {}
        self.class_to_info = {}
        
        count = 0
        for cls in self.onto.classes():
            # Get the class name (IRI fragment)
            class_name = cls.name if hasattr(cls, 'name') else str(cls).split('#')[-1]
            
            # Try to get codigoTerminologia property
            codigo = None
            nombre = None
            
            # Check for codigoTerminologia annotation - SAFE access
            try:
                codigos_prop = getattr(cls, 'codigoTerminologia', None)
                if codigos_prop is not None:
                    codigos = list(codigos_prop)
                    if codigos:
                        codigo = str(codigos[0])
            except (TypeError, AttributeError):
                pass
            
            # Check for nombreLocal annotation - SAFE access
            try:
                nombres_prop = getattr(cls, 'nombreLocal', None)
                if nombres_prop is not None:
                    nombres = list(nombres_prop)
                    if nombres:
                        nombre = str(nombres[0])
            except (TypeError, AttributeError):
                pass
            
            # Store class info
            self.class_to_info[cls] = {
                'name': class_name,
                'code': codigo,
                'local_name': nombre or class_name
            }
            
            # Map code to class
            if codigo:
                self.code_to_class[codigo] = cls
                count += 1
            
            # Also map by class name (IRI fragment) for RID-style codes
            self.code_to_class[class_name] = cls
        
        print(f"[OK] Built lookup for {count} coded classes, {len(self.code_to_class)} total mappings")
    
    def _get_class_label(self, cls) -> str:
        """Get a readable label for a class."""
        info = self.class_to_info.get(cls, {})
        return info.get('local_name', cls.name if hasattr(cls, 'name') else str(cls).split('#')[-1])
    
    def get_lineage(self, code: str) -> List[Tuple[str, str]]:
        """
        Get the lineage (path to root) for a given code.
        
        Args:
            code: Entity code (e.g., "415582006" or "RID666")
            
        Returns:
            List of (class_name, label) tuples from root to leaf
        """
        # Find the class
        cls = self.code_to_class.get(code)
        
        if cls is None:
            # Try case-insensitive match
            for key, val in self.code_to_class.items():
                if str(key).lower() == code.lower():
                    cls = val
                    break
        
        if cls is None:
            return [("Unknown", f"[{code}] (not in ontology)")]
        
        # Build lineage by walking up is_a relationships
        lineage = []
        visited = set()
        current = cls
        
        while current is not None and current not in visited:
            visited.add(current)
            
            class_name = current.name if hasattr(current, 'name') else str(current).split('#')[-1]
            label = self._get_class_label(current)
            lineage.append((class_name, label))
            
            # Get parent(s) - filter out Thing and non-class parents
            try:
                parents = [p for p in current.is_a if p != Thing and hasattr(p, 'name')]
                current = parents[0] if parents else None
            except:
                current = None
        
        # Reverse to get root -> leaf order
        lineage.reverse()
        return lineage
    
    def build_tree_for_note(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a hierarchical tree for a medical note.
        
        Args:
            note: Note dict with 'id' and 'extracted_entities'
            
        Returns:
            Hierarchical tree structure
        """
        note_id = note.get('id', 'unknown')
        entities = note.get('extracted_entities', [])
        
        # Group by field_location first
        by_field = defaultdict(list)
        for ent in entities:
            field = ent.get('field_location', 'unknown')
            by_field[field].append(ent)
        
        # Build tree for each field
        tree = {
            'note_id': note_id,
            'fields': {}
        }
        
        for field, field_entities in by_field.items():
            field_tree = self._build_field_tree(field_entities)
            tree['fields'][field] = field_tree
        
        return tree
    
    def _build_field_tree(self, entities: List[Dict]) -> Dict:
        """Build a tree structure for entities in a single field."""
        # Build a nested dict representing the tree
        root = {}
        
        for ent in entities:
            code = ent.get('code', 'unknown')
            text = ent.get('text', '')
            category = ent.get('category', '')
            source = ent.get('source', '')
            
            # Get lineage
            lineage = self.get_lineage(code)
            
            # Insert into tree
            current = root
            for i, (class_name, label) in enumerate(lineage):
                if class_name not in current:
                    current[class_name] = {
                        '_label': label,
                        '_entities': [],
                        '_children': {}
                    }
                
                # If this is the leaf, add the entity
                if i == len(lineage) - 1:
                    current[class_name]['_entities'].append({
                        'code': code,
                        'text': text,
                        'category': category,
                        'source': source
                    })
                
                current = current[class_name]['_children']
        
        return root
    
    def print_ascii_tree(self, tree: Dict[str, Any], max_depth: int = 10):
        """
        Print a visual ASCII tree representation.
        
        Args:
            tree: Tree structure from build_tree_for_note()
            max_depth: Maximum depth to display
        """
        note_id = tree.get('note_id', 'unknown')
        fields = tree.get('fields', {})
        
        print(f"\n{'='*60}")
        print(f"Note: {note_id}")
        print('='*60)
        
        for field_name, field_tree in fields.items():
            print(f"\n[{field_name.upper()}]")
            self._print_node(field_tree, prefix="", depth=0, max_depth=max_depth)
    
    def _print_node(self, node: Dict, prefix: str, depth: int, max_depth: int):
        """Recursively print tree nodes."""
        if depth > max_depth:
            return
        
        items = list(node.items())
        for i, (class_name, data) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "|__ " if is_last else "|-- "
            
            label = data.get('_label', class_name)
            entities = data.get('_entities', [])
            children = data.get('_children', {})
            
            # Print this node
            if entities:
                for ent in entities:
                    text = ent.get('text', '')
                    code = ent.get('code', '')
                    source = ent.get('source', '')
                    print(f"{prefix}{connector}{label} -> [{code}] \"{text}\" ({source})")
            else:
                print(f"{prefix}{connector}{label}")
            
            # Print children
            new_prefix = prefix + ("    " if is_last else "|   ")
            if children:
                self._print_node(children, new_prefix, depth + 1, max_depth)
    
    def process_all_notes(self, results_path: str, output_path: str = None) -> List[Dict]:
        """
        Process all notes from ensemble results.
        
        Args:
            results_path: Path to ensemble_results.json
            output_path: Optional path to save knowledge graph JSON
            
        Returns:
            List of tree structures for all notes
        """
        print(f"\n[Processing] Loading entities from: {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            notes = json.load(f)
        
        print(f"[OK] Loaded {len(notes)} notes")
        
        trees = []
        for i, note in enumerate(notes):
            tree = self.build_tree_for_note(note)
            trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(notes)} notes...")
        
        print(f"[OK] Built trees for {len(trees)} notes")
        
        # Save to JSON if output path provided
        return trees
    
    def export_to_html(self, trees: List[Dict], output_path: str, title: str = "Ontology Knowledge Graph"):
        """
        Export trees to an interactive HTML visualization.
        
        Args:
            trees: List of tree structures from process_all_notes()
            output_path: Path to save the HTML file
            title: Page title
        """
        html_content = self._generate_html(trees, title)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[SAVED] HTML visualization saved to: {output_path}")
    
    def _generate_html(self, trees: List[Dict], title: str) -> str:
        """Generate the complete HTML document."""
        # Generate tree HTML for each note
        notes_html = []
        for tree in trees:
            note_html = self._generate_note_html(tree)
            notes_html.append(note_html)
        
        all_notes_html = "\n".join(notes_html)
        
        html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2937;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-orange: #f59e0b;
            --accent-red: #ef4444;
            --accent-cyan: #06b6d4;
            --border-color: #374151;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        
        .controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            justify-content: center;
        }}
        
        .search-box {{
            flex: 1;
            max-width: 400px;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-card);
            color: var(--text-primary);
            font-size: 1rem;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }}
        
        .btn {{
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }}
        
        .btn-secondary {{
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }}
        
        .note-card {{
            background: var(--bg-card);
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .note-card:hover {{
            border-color: var(--accent-blue);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        .note-header {{
            padding: 1rem 1.5rem;
            background: rgba(255,255,255,0.03);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .note-header:hover {{
            background: rgba(255,255,255,0.06);
        }}
        
        .note-id {{
            font-family: 'Consolas', monospace;
            color: var(--accent-cyan);
            font-size: 0.9rem;
        }}
        
        .note-toggle {{
            font-size: 1.2rem;
            transition: transform 0.3s ease;
        }}
        
        .note-card.collapsed .note-toggle {{
            transform: rotate(-90deg);
        }}
        
        .note-card.collapsed .note-body {{
            display: none;
        }}
        
        .note-body {{
            padding: 1rem 1.5rem;
        }}
        
        .field-section {{
            margin-bottom: 1rem;
        }}
        
        .field-header {{
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            display: inline-block;
        }}
        
        .field-history {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-orange); }}
        .field-findings {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }}
        .field-impression {{ background: rgba(139, 92, 246, 0.2); color: var(--accent-purple); }}
        .field-unknown {{ background: rgba(148, 163, 184, 0.2); color: var(--text-secondary); }}
        
        .tree {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            line-height: 1.8;
        }}
        
        .tree-node {{
            padding-left: 1.5rem;
            border-left: 1px dashed var(--border-color);
            margin-left: 0.5rem;
        }}
        
        .tree-root > .tree-node {{
            border-left: none;
            padding-left: 0;
            margin-left: 0;
        }}
        
        .node-label {{
            cursor: pointer;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: background 0.2s ease;
            display: inline-block;
        }}
        
        .node-label:hover {{
            background: rgba(255,255,255,0.1);
        }}
        
        .node-expanded > .node-label::before {{
            content: '▼ ';
            color: var(--accent-blue);
            font-size: 0.7rem;
        }}
        
        .node-collapsed > .node-label::before {{
            content: '▶ ';
            color: var(--accent-blue);
            font-size: 0.7rem;
        }}
        
        .node-leaf > .node-label::before {{
            content: '• ';
            color: var(--accent-green);
        }}
        
        .node-collapsed > .tree-node {{
            display: none;
        }}
        
        .entity {{
            background: rgba(59, 130, 246, 0.15);
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            margin: 0.2rem 0;
            display: inline-block;
            border-left: 3px solid var(--accent-blue);
        }}
        
        .entity-code {{
            color: var(--accent-cyan);
            font-weight: 600;
        }}
        
        .entity-text {{
            color: var(--text-primary);
        }}
        
        .entity-source {{
            font-size: 0.75rem;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            margin-left: 0.5rem;
        }}
        
        .source-llm {{ background: var(--accent-purple); color: white; }}
        .source-dfa {{ background: var(--accent-orange); color: white; }}
        .source-other {{ background: var(--text-secondary); color: white; }}
        
        .category-tag {{
            font-size: 0.7rem;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            margin-left: 0.3rem;
            background: rgba(255,255,255,0.1);
        }}
        
        .unknown-warning {{
            color: var(--accent-orange);
            font-style: italic;
        }}
        
        .stats {{
            display: flex;
            gap: 2rem;
            justify-content: center;
            margin-top: 1rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .stat {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .stat-value {{
            color: var(--accent-blue);
            font-weight: 600;
        }}
        
        .highlight {{
            background: rgba(245, 158, 11, 0.4) !important;
            border-radius: 2px;
        }}
        
        /* Visual Graph Styles */
        .view-tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .view-tab {{
            padding: 0.5rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }}
        
        .view-tab:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .view-tab.active {{
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-blue);
        }}
        
        .graph-container {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
            overflow-x: auto;
            min-height: 400px;
        }}
        
        .graph-container svg {{
            display: block;
            margin: 0 auto;
        }}
        
        .graph-container .node circle {{
            stroke: var(--accent-blue);
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .graph-container .node circle:hover {{
            stroke-width: 3px;
            filter: drop-shadow(0 0 6px var(--accent-blue));
        }}
        
        .graph-container .node text {{
            font-size: 11px;
            fill: var(--text-primary);
            font-family: 'Segoe UI', sans-serif;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
        }}
        
        .graph-container .link {{
            fill: none;
            stroke: var(--border-color);
            stroke-width: 1.5px;
            opacity: 0.6;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            h1 {{ font-size: 1.8rem; }}
            .controls {{ flex-direction: column; }}
            .search-box {{ max-width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 {title}</h1>
            <p class="subtitle">Interactive Ontology-based Entity Hierarchy</p>
            <div class="stats">
                <div class="stat"><span>Notes:</span> <span class="stat-value">{len(trees)}</span></div>
            </div>
        </header>
        
        <div class="controls">
            <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search entities, codes, or concepts...">
            <button class="btn btn-primary" onclick="expandAll()">Expand All</button>
            <button class="btn btn-secondary" onclick="collapseAll()">Collapse All</button>
        </div>
        
        <div id="notesContainer">
            {all_notes_html}
        </div>
    </div>
    
    <script>
        // Toggle note card
        function toggleNote(noteId) {{
            const card = document.getElementById('note-' + noteId);
            card.classList.toggle('collapsed');
        }}
        
        // Toggle tree node
        function toggleNode(elem) {{
            const parent = elem.parentElement;
            if (parent.classList.contains('node-expanded')) {{
                parent.classList.remove('node-expanded');
                parent.classList.add('node-collapsed');
            }} else if (parent.classList.contains('node-collapsed')) {{
                parent.classList.remove('node-collapsed');
                parent.classList.add('node-expanded');
            }}
        }}
        
        // Switch between Tree and Graph views
        function switchView(safeId, viewType) {{
            // Using parent element query selector
            const cardId = 'note-' + safeId;
            const noteCard = document.getElementById(cardId);
            const container = noteCard.querySelector('.note-body');
            
            // Update tabs
            const tabs = container.querySelectorAll('.view-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Find clicked button (simple approach)
            if (viewType === 'tree') tabs[0].classList.add('active');
            else tabs[1].classList.add('active');
            
            // Update content
            const treeView = document.getElementById('tree-' + safeId);
            const graphView = document.getElementById('graph-' + safeId);
            
            if (viewType === 'tree') {{
                treeView.classList.add('active');
                graphView.classList.remove('active');
            }} else {{
                treeView.classList.remove('active');
                graphView.classList.add('active');
                
                // Render graph if empty
                const graphContainer = document.getElementById('graph-container-' + safeId);
                if (!graphContainer.hasChildNodes()) {{
                    renderGraph(safeId);
                }}
            }}
        }}
        
        // Render D3 Graph
        function renderGraph(safeId) {{
            const data = window['treeData_' + safeId];
            if (!data) return;
            
            const width = 800;
            const container = document.getElementById('graph-container-' + safeId);
            const height = Math.max(400, data.children.length * 100); // Dynamic height
            
            // Clear previous
            container.innerHTML = '';
            
            const svg = d3.select(container).append("svg")
                .attr("width", "100%")
                .attr("height", height)
                .attr("viewBox", [0, 0, width, height])
                .style("font", "10px sans-serif")
                .style("user-select", "none");
                
            const g = svg.append("g")
                .attr("transform", "translate(100,0)");
                
            const treeLayout = d3.tree().size([height, width - 200]);
            
            const root = d3.hierarchy(data);
            treeLayout(root);
            
            // Links
            g.append("g")
                .attr("fill", "none")
                .attr("stroke", "#555")
                .attr("stroke-opacity", 0.4)
                .attr("stroke-width", 1.5)
              .selectAll("path")
              .data(root.links())
              .join("path")
                .attr("d", d3.linkHorizontal()
                    .x(d => d.y)
                    .y(d => d.x));
            
            // Nodes
            const node = g.append("g")
              .selectAll("g")
              .data(root.descendants())
              .join("g")
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`);
            
            node.append("circle")
                .attr("r", 6)
                .attr("fill", d => {{
                    if (d.data.type === 'field') return "#3b82f6";
                    if (d.data.type === 'anatomy') return "#10b981";
                    if (d.data.type === 'finding') return "#8b5cf6";
                    if (d.data.type === 'unknown') return "#f59e0b";
                    return "#999";
                }})
                .attr("stroke-width", 2)
                .attr("stroke", "#fff");
            
            node.append("text")
                .attr("dy", "0.31em")
                .attr("x", d => d.children ? -8 : 8)
                .attr("text-anchor", d => d.children ? "end" : "start")
                .text(d => d.data.name)
                .clone(true).lower()
                .attr("stroke", "white");
                
            // Add zoom capability
            const zoom = d3.zoom()
                .scaleExtent([0.5, 3])
                .on("zoom", (event) => {{
                    g.attr("transform", event.transform);
                }});
                
            svg.call(zoom)
               .call(zoom.transform, d3.zoomIdentity.translate(100, 0));
        }}
        
        // Expand all
        function expandAll() {{
            document.querySelectorAll('.note-card').forEach(card => card.classList.remove('collapsed'));
            document.querySelectorAll('.node-collapsed').forEach(node => {{
                node.classList.remove('node-collapsed');
                node.classList.add('node-expanded');
            }});
        }}
        
        // Collapse all
        function collapseAll() {{
            document.querySelectorAll('.note-card').forEach(card => card.classList.add('collapsed'));
            document.querySelectorAll('.node-expanded').forEach(node => {{
                node.classList.remove('node-expanded');
                node.classList.add('node-collapsed');
            }});
        }}
        
        // Search functionality
        document.getElementById('searchBox').addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase().trim();
            
            // Remove existing highlights
            document.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
            
            if (!query) {{
                document.querySelectorAll('.note-card').forEach(card => card.style.display = '');
                return;
            }}
            
            document.querySelectorAll('.note-card').forEach(card => {{
                const text = card.textContent.toLowerCase();
                if (text.includes(query)) {{
                    card.style.display = '';
                    card.classList.remove('collapsed');
                    
                    // Highlight matches
                    card.querySelectorAll('.entity, .node-label').forEach(el => {{
                        if (el.textContent.toLowerCase().includes(query)) {{
                            el.classList.add('highlight');
                            // Expand parent nodes
                            let parent = el.parentElement;
                            while (parent) {{
                                if (parent.classList.contains('node-collapsed')) {{
                                    parent.classList.remove('node-collapsed');
                                    parent.classList.add('node-expanded');
                                }}
                                parent = parent.parentElement;
                            }}
                        }}
                    }});
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }});
        
        // Start collapsed
        document.addEventListener('DOMContentLoaded', function() {{
            collapseAll();
            // Expand first note
            const firstNote = document.querySelector('.note-card');
            if (firstNote) firstNote.classList.remove('collapsed');
        }});
    </script>
</body>
</html>'''
        return html
    
    def _generate_note_html(self, tree: Dict) -> str:
        """Generate HTML for a single note."""
        note_id = tree.get('note_id', 'unknown')
        fields = tree.get('fields', {})
        
        # Generate unique safe ID for JavaScript
        safe_id = note_id.replace('-', '_').replace(' ', '_')
        
        fields_html = []
        for field_name, field_tree in fields.items():
            field_class = f"field-{field_name.lower()}" if field_name.lower() in ['history', 'findings', 'impression'] else 'field-unknown'
            tree_html = self._generate_tree_html(field_tree)
            
            fields_html.append(f'''
            <div class="field-section">
                <div class="field-header {field_class}">{field_name.upper()}</div>
                <div class="tree tree-root">
                    {tree_html}
                </div>
            </div>
            ''')
        
        all_fields_html = "\n".join(fields_html)
        
        # Generate D3 tree data for all fields combined
        d3_tree_data = self._generate_d3_tree_data(fields)
        d3_tree_json = json.dumps(d3_tree_data, ensure_ascii=False)
        
        return f'''
        <div class="note-card" id="note-{note_id.replace('-', '_').replace(' ', '_')}">
            <div class="note-header" onclick="toggleNote('{note_id.replace('-', '_').replace(' ', '_')}')">
                <span class="note-id">📋 Note: {note_id}</span>
                <span class="note-toggle">▼</span>
            </div>
            <div class="note-body">
                <div class="view-tabs">
                    <button class="view-tab active" onclick="switchView('{safe_id}', 'tree')">📝 Text Tree</button>
                    <button class="view-tab" onclick="switchView('{safe_id}', 'graph')">🌳 Visual Graph</button>
                </div>
                <div class="tree-view active" id="tree-{safe_id}">
                    {all_fields_html}
                </div>
                <div class="graph-view" id="graph-{safe_id}">
                    <div class="graph-container" id="graph-container-{safe_id}"></div>
                </div>
                <script>
                    window.treeData_{safe_id} = {d3_tree_json};
                </script>
            </div>
        </div>
        '''
    
    def _generate_d3_tree_data(self, fields: Dict) -> Dict:
        """Generate D3-compatible hierarchical tree data from fields."""
        root = {
            "name": "Note",
            "children": []
        }
        
        for field_name, field_tree in fields.items():
            field_node = {
                "name": field_name.upper(),
                "type": "field",
                "children": self._convert_tree_to_d3(field_tree)
            }
            root["children"].append(field_node)
        
        return root
    
    def _convert_tree_to_d3(self, node: Dict) -> List[Dict]:
        """Recursively convert tree structure to D3 format."""
        result = []
        
        for class_name, data in node.items():
            label = data.get('_label', class_name)
            entities = data.get('_entities', [])
            children = data.get('_children', {})
            
            # Determine node type based on entities
            node_type = "category"
            if entities:
                categories = [e.get('category', '') for e in entities]
                if 'ANATOMY' in categories:
                    node_type = "anatomy"
                elif 'FINDING' in categories:
                    node_type = "finding"
                elif label.startswith('[') and 'not in ontology' in label:
                    node_type = "unknown"
            
            d3_node = {
                "name": label[:25] + "..." if len(label) > 25 else label,
                "fullName": label,
                "type": node_type,
                "entities": entities,
                "children": self._convert_tree_to_d3(children)
            }
            
            result.append(d3_node)
        
        return result
    
    def _generate_tree_html(self, node: Dict, depth: int = 0) -> str:
        """Recursively generate HTML for tree nodes."""
        if not node:
            return ""
        
        html_parts = []
        
        for class_name, data in node.items():
            label = data.get('_label', class_name)
            entities = data.get('_entities', [])
            children = data.get('_children', {})
            
            has_children = bool(children)
            node_class = "node-expanded" if has_children else "node-leaf"
            
            # Generate entities HTML
            entities_html = ""
            if entities:
                for ent in entities:
                    code = ent.get('code', '')
                    text = ent.get('text', '')
                    source = ent.get('source', '')
                    category = ent.get('category', '')
                    
                    source_class = f"source-{source.lower()}" if source.lower() in ['llm', 'dfa'] else 'source-other'
                    cat_html = f'<span class="category-tag">{category}</span>' if category else ''
                    
                    entities_html += f'''
                    <div class="entity">
                        <span class="entity-code">[{code}]</span>
                        <span class="entity-text">"{text}"</span>
                        <span class="entity-source {source_class}">{source}</span>
                        {cat_html}
                    </div>
                    '''
            
            # Generate children HTML
            children_html = self._generate_tree_html(children, depth + 1) if children else ""
            
            # Add warning for unknown entities
            label_class = "unknown-warning" if label.startswith('[') and 'not in ontology' in label else ""
            
            html_parts.append(f'''
            <div class="{node_class}">
                <span class="node-label {label_class}" onclick="toggleNode(this)">{label}</span>
                {entities_html}
                <div class="tree-node">
                    {children_html}
                </div>
            </div>
            ''')
        
        return "\n".join(html_parts)
    
    def process_all_notes(self, results_path: str, output_path: str = None) -> List[Dict]:
        """
        Process all notes from ensemble results.
        
        Args:
            results_path: Path to ensemble_results.json
            output_path: Optional path to save knowledge graph JSON
            
        Returns:
            List of tree structures for all notes
        """
        print(f"\n[Processing] Loading entities from: {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            notes = json.load(f)
        
        print(f"[OK] Loaded {len(notes)} notes")
        
        trees = []
        for i, note in enumerate(notes):
            tree = self.build_tree_for_note(note)
            trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(notes)} notes...")
        
        print(f"[OK] Built trees for {len(trees)} notes")
        
        # Save to JSON if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(trees, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] Knowledge graph saved to: {output_path}")
        
        return trees


# --- Main Entry Point ---
if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OWL_FILE = os.path.join(BASE_DIR, "data", "llm_snomed_final4.owl")
    RESULTS_FILE = os.path.join(BASE_DIR, "data", "processed", "ensemble_results.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "knowledge_graph.json")
    HTML_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "knowledge_graph.html")

    try:
        # Initialize tree builder
        tree_builder = OntologyTree(OWL_FILE)
        
        # Process all notes
        trees = tree_builder.process_all_notes(RESULTS_FILE, OUTPUT_FILE)
        
        # Export to HTML
        tree_builder.export_to_html(trees, HTML_OUTPUT_FILE, "Radiology NER Knowledge Graph")
        
        # Print sample trees
        print("\n" + "="*60)
        print("SAMPLE OUTPUT: First 2 Notes")
        print("="*60)
        
        for tree in trees[:2]:
            tree_builder.print_ascii_tree(tree)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
