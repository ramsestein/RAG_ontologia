"""
RAG Augmentation Module - Query expansion and bilingual hints
Enhances queries with EN↔ES translations and domain-specific expansions
"""

import os
import json
from typing import List, Dict, Iterable


class QueryAugmenter:
    """Gestiona expansión de queries y hints bilingües"""
    
    def __init__(self, assets_dir: str):
        """
        Args:
            assets_dir: Directorio donde buscar bilingual_hints.json
        """
        self.assets_dir = assets_dir
        self.hints: Dict[str, Iterable[str]] = {}
        self._load_bilingual_hints()
    
    def _load_bilingual_hints(self):
        """
        Carga mapeos EN→ES desde JSON opcional (assets/bilingual_hints.json).
        Si no existe, usa un conjunto por defecto.
        """
        default_hints = {
            "stroke": ["ictus", "accidente cerebrovascular"],
            "headache": ["cefalea", "dolor de cabeza"],
            "nausea": ["náusea"],
            "vomiting": ["vómito", "emesis"],
            "weakness": ["debilidad", "astenia"],
            "hemiparesis": ["hemiparesia"],
            "hemiplegia": ["hemiplejia"],
            "aphasia": ["afasia"],
            "dysarthria": ["disartria"],
            "atrial fibrillation": ["fibrilación auricular", "FA"],
            "thrombectomy": ["trombectomía", "terapia endovascular"],
            "tpa": ["alteplasa", "tPA", "rtPA"],
            "middle cerebral artery": ["arteria cerebral media", "ACM"],
            "mca": ["arteria cerebral media"],
            "occlusion": ["oclusión"],
            "recanalization": ["recanalización"],
            "ischemic penumbra": ["penumbra isquémica", "penumbra"],
            "hemorrhage": ["hemorragia", "hemorragia intracerebral", "hemorragia intracraneal"],
            "infarct": ["infarto cerebral", "infarto isquémico"],
            "nihss": ["escala de ictus nihssis", "escala nihss", "escala de ictus del nih"],
            "aspects": ["escala aspects", "puntuación aspects", "escala de aspectos"],
            "tici": ["escala tici", "grado tici"],
            "gcs": ["escala glasgow", "glasgow coma scale"],
            "mrs": ["escala de rankin modificada", "modified rankin scale"],
            "mca": ["arteria cerebral media", "acm", "m1", "m2"],
            "ica": ["arteria carótida interna", "aci"],
            "basilar": ["arteria basilar"],
            "pica": ["arteria cerebelosa posteroinferior"],
            "aica": ["arteria cerebelosa anteroinferior"],
            "sca": ["arteria cerebelosa superior"],
            "thrombectomy": ["trombectomía", "tratamiento endovascular"],
            "coiling": ["embolización con coils", "coils endovasculares"],
            "angioplasty": ["angioplastia"],
            "endarterectomy": ["endarterectomía"],
            "occlusion": ["oclusión"],
            "stenosis": ["estenosis"],
            "hemorrhage": ["hemorragia", "sangrado"],
            "infarct": ["infarto cerebral", "infarto isquémico"]
        }
        path = os.path.join(self.assets_dir, "bilingual_hints.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # normalizar a listas
                for k, v in data.items():
                    if isinstance(v, str):
                        data[k.lower()] = [v]
                    elif isinstance(v, list):
                        data[k.lower()] = list({str(x) for x in v if str(x).strip()})
                self.hints = {**default_hints, **data}
                print(f"[AUGMENTATION] [OK] Hints bilingües cargados desde {path} ({len(self.hints)} entradas)")
            else:
                self.hints = default_hints
                print(f"[AUGMENTATION] [OK] Hints bilingües por defecto ({len(self.hints)} entradas)")
        except Exception as e:
            print(f"[AUGMENTATION] [WARNING] No se pudieron cargar hints desde JSON: {e}")
            self.hints = default_hints
    
    def expand_query_variants(self, query: str) -> List[str]:
        """
        Genera variantes de búsqueda:
        - original
        - lower
        - hints EN→ES si existen
        """
        variants = []
        if not isinstance(query, str):
            query = str(query)
        base = query.strip()
        if not base:
            return variants
        candidates = [base, base.lower()]
        key = base.lower()
        if key in self.hints:
            # puede ser lista
            for es in self.hints[key]:
                candidates.append(es)
        # deduplicación conservando orden
        seen = set()
        for c in candidates:
            c2 = c.strip()
            if c2 and c2 not in seen:
                seen.add(c2)
                variants.append(c2)
        return variants
