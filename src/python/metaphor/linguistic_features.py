"""
Extracción de Características Lingüísticas para Detección de Metáforas:
1. Índices de Abstracción y Concreción (Brysbaert et al. y morfología derivativa).
2. WordNet & VerbNet:
   - Desambiguación léxica de synsets (sustantivos, verbos, adjetivos).
   - Detección de dominios semánticos (Supersenses / Lexicographer files).
   - Incongruencia de dominio (Frame Clash / Domain Disparity) inspirado en IlliniMet y MetaNet.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

try:
    from nltk.corpus import wordnet as wn
    NLTK_WORDNET_AVAILABLE = True
except Exception:
    NLTK_WORDNET_AVAILABLE = False


class LinguisticFeatureExtractor:
    """
    Extrae rasgos lingüísticos esenciales para la detección de metáforas:
    - Nivel de concreción vs abstracción de cada palabra.
    - Dominios semánticos extraídos de WordNet (dominio fuente vs dominio destino).
    """

    # Léxico base de concreción (1.0 = abstracto, 5.0 = concreto)
    CONCRETENESS_LEXICON = {
        # Términos altamente concretos (entidades físicas, órganos, objetos, movimiento físico)
        "árbol": 4.9, "piedra": 4.9, "agua": 4.9, "camino": 4.6, "puente": 4.7, "semilla": 4.8,
        "raíz": 4.7, "raíces": 4.7, "fruto": 4.8, "frutos": 4.8, "casa": 4.8, "edificio": 4.7,
        "muro": 4.8, "muros": 4.8, "barrera": 4.4, "cimientos": 4.7, "pilar": 4.6, "pilares": 4.6,
        "mano": 4.9, "corazón": 4.6, "luz": 4.3, "fuego": 4.8, "pesado": 4.2, "volar": 4.5,
        "correr": 4.7, "construir": 4.5, "edificar": 4.5, "levantar": 4.4, "sembrar": 4.6,
        "cosechar": 4.6, "apagar": 4.4, "trinchera": 4.6, "armas": 4.7, "batalla": 4.3,
        "sendero": 4.5, "escalera": 4.8, "escaleras": 4.8,
        "tree": 4.9, "stone": 4.9, "water": 4.9, "bridge": 4.7, "seed": 4.8, "root": 4.7,
        "wall": 4.8, "fire": 4.8, "fly": 4.5, "run": 4.7, "build": 4.5,

        # Términos altamente abstractos (conceptos, valores, tiempo, cognición)
        "tiempo": 1.7, "idea": 1.6, "ideas": 1.6, "pensamiento": 1.8, "esperanza": 1.5, "libertad": 1.4,
        "justicia": 1.3, "verdad": 1.4, "conocimiento": 1.7, "futuro": 1.8, "destino": 1.4,
        "amor": 1.9, "miedo": 2.1, "tristeza": 1.9, "problema": 2.0, "desafío": 1.9,
        "cultura": 1.8, "sociedad": 2.1, "educación": 2.2, "progreso": 1.7, "éxito": 1.8,
        "democracia": 1.4, "paz": 1.5, "corrupción": 1.6, "impunidad": 1.5, "soberanía": 1.4,
        "debate": 1.8, "bienestar": 1.6, "compromiso": 1.6, "confianza": 1.5, "valores": 1.5,
        "time": 1.7, "idea": 1.6, "hope": 1.5, "freedom": 1.4, "justice": 1.3, "knowledge": 1.7
    }

    # Sufijos morfológicos típicamente abstractos en español
    ABSTRACT_SUFFIXES = ("ción", "cion", "sión", "sion", "dad", "tad", "ismo", "eza", "ura", "miento", "ancia", "encia")

    def __init__(self):
        self.wn = wn if NLTK_WORDNET_AVAILABLE else None

    def get_concreteness(self, word: str) -> float:
        """
        Calcula el valor de concreción de la palabra entre 1.0 (abstracto) y 5.0 (concreto).
        Utiliza el léxico calibrado y reglas morfológicas derivativas.
        """
        w = word.lower().strip()
        if w in self.CONCRETENESS_LEXICON:
            return self.CONCRETENESS_LEXICON[w]

        # Regla morfológica: sufijos nominalizadores abstractos
        for suffix in self.ABSTRACT_SUFFIXES:
            if w.endswith(suffix):
                return 1.8

        # Valor neutro medio por defecto
        return 3.0

    def get_wordnet_domains(self, word: str, pos: Optional[str] = None) -> List[str]:
        """
        Extrae los dominios léxicos (supersenses / lexnames) de WordNet para la palabra.
        Ejemplos: 'noun.cognition', 'noun.artifact', 'verb.motion', 'verb.creation', etc.
        """
        if not self.wn:
            return ["unknown"]

        w = word.lower().strip()
        synsets = []

        try:
            # Buscar en español
            synsets = self.wn.synsets(w, lang="spa")
            if not synsets:
                # Buscar en inglés
                synsets = self.wn.synsets(w)
        except Exception:
            pass

        if not synsets:
            return ["unknown"]

        # Filtrar o priorizar según categoría sintáctica
        domains = []
        for s in synsets[:3]:  # Tomar los sentidos más frecuentes
            lexname = s.lexname()  # ej. 'noun.time', 'verb.motion'
            domains.append(lexname)

        return domains if domains else ["unknown"]

    def calculate_domain_disparity(self, word_a: str, word_b: str) -> float:
        """
        Calcula la disparidad conceptual entre dos palabras en una relación sintáctica
        (ej. Sujeto-Verbo o Verbo-Objeto).
        En la teoría de la metáfora conceptual (MetaNet / Lakoff), una metáfora surge cuando
        un dominio físico/concreto proyecta estructura sobre un dominio abstracto/cognitivo.
        """
        conc_a = self.get_concreteness(word_a)
        conc_b = self.get_concreteness(word_b)
        concreteness_gap = abs(conc_a - conc_b)

        domains_a = self.get_wordnet_domains(word_a)
        domains_b = self.get_wordnet_domains(word_b)

        # Verificar si hay choque de dominios (ej. 'verb.motion' con 'noun.cognition' o 'noun.time')
        is_motion = any("motion" in d or "contact" in d or "body" in d for d in (domains_a + domains_b))
        is_abstract = any("cognition" in d or "feeling" in d or "time" in d or "attribute" in d for d in (domains_a + domains_b))

        frame_clash = 1.0 if (is_motion and is_abstract) else 0.0

        # Normalizar disparidad a escala [0, 1]
        disparity = (concreteness_gap / 3.5) * 0.65 + frame_clash * 0.35
        return float(min(1.0, max(0.0, disparity)))

    def extract_token_features(self, word: str, pos_tag: str = "") -> List[float]:
        """
        Construye un vector denso de características lingüísticas para enriquecer embeddings:
        [concreteness_norm, is_abstract, is_concrete, is_verb, is_noun, is_adj, disparity_indicator]
        """
        conc = self.get_concreteness(word)
        conc_norm = (conc - 1.0) / 4.0  # Rango [0.0, 1.0]
        is_abstract = 1.0 if conc < 2.5 else 0.0
        is_concrete = 1.0 if conc > 3.8 else 0.0

        pos_upper = pos_tag.upper() if pos_tag else ""
        is_verb = 1.0 if pos_upper.startswith("VB") or "VERB" in pos_upper else 0.0
        is_noun = 1.0 if pos_upper.startswith("NN") or "NOUN" in pos_upper else 0.0
        is_adj = 1.0 if pos_upper.startswith("JJ") or "ADJ" in pos_upper else 0.0

        return [conc_norm, is_abstract, is_concrete, is_verb, is_noun, is_adj]
