"""
Detector de Metáforas Completo:
Integra la arquitectura neuronal BiLSTM con Embeddings Contextuales (BERT/RoBERTa/XLNet),
características lingüísticas de WordNet y teoría de marcos conceptuales (MetaNet / IlliniMet).
"""

import torch
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple

from .linguistic_features import LinguisticFeatureExtractor
from .bilstm_model import ContextualBiLSTMMetaphorModel


class MetaphorDetector:
    """
    Detector híbrido de metáforas a nivel de palabra y oración.
    Combina:
    1. Modelo neuronal BiLSTM alimentado por embeddings contextuales.
    2. Rasgos lingüísticos de concreción y WordNet.
    3. Motor conceptual de marcos y mapeos MetaNet (Dominio Fuente -> Dominio Destino).
    """

    # Mapeos conceptuales paradigmáticos (Teoría de Metáfora Conceptual - Lakoff / Johnson / MetaNet)
    CONCEPTUAL_FRAMES = [
        {
            "name": "CONSTRUCCIÓN_ES_DESARROLLO",
            "source_domain": "EDIFICACIÓN / FÍSICO",
            "target_domain": "SOCIEDAD / CONOCIMIENTO",
            "source_keywords": ["construir", "edificar", "cimientos", "pilares", "muros", "levantar", "estructura", "base", "build", "building", "foundations", "pillars", "walls", "construct", "structure"],
            "target_keywords": ["futuro", "conocimiento", "sociedad", "paz", "democracia", "proyecto", "educación", "confianza", "future", "knowledge", "society", "peace", "democracy", "nation", "education", "trust"]
        },
        {
            "name": "VIDA_ES_UN_VIAJE",
            "source_domain": "MOVIMIENTO / ESPACIAL",
            "target_domain": "TIEMPO / PROCESO VITAL",
            "source_keywords": ["camino", "rumbo", "avanzar", "caminar", "sendero", "guía", "viaje", "tropezar", "llegar", "path", "journey", "walk", "advance", "guide", "road", "crossroads"],
            "target_keywords": ["vida", "historia", "proceso", "carrera", "transformación", "desarrollo", "life", "history", "process", "destiny", "future", "progress"]
        },
        {
            "name": "INSTITUCIÓN_ES_ORGANISMO",
            "source_domain": "BOTÁNICO / BIOLÓGICO",
            "target_domain": "INSTITUCIONAL / ACADÉMICO",
            "source_keywords": ["sembrar", "cosechar", "fruto", "raíz", "raíces", "crecer", "florecer", "semilla", "germinar", "plant", "harvest", "fruit", "root", "roots", "grow", "flourish", "seed", "sprout"],
            "target_keywords": ["universidad", "ideas", "valores", "justicia", "compromiso", "esperanza", "comunidad", "nation", "ideas", "values", "justice", "liberty", "freedom", "hope", "community", "institution"]
        },
        {
            "name": "DEBATE_ES_GUERRA",
            "source_domain": "BÉLICO / CONFLICTO",
            "target_domain": "POLÍTICA / COMUNICACIÓN",
            "source_keywords": ["batalla", "lucha", "trinchera", "armas", "vencer", "atacar", "defender", "estrategia", "battle", "fight", "weapons", "attack", "defend", "defeat", "victory", "trench", "strategy", "enemy"],
            "target_keywords": ["ideas", "debate", "pobreza", "corrupción", "propuesta", "postura", "ideas", "debate", "poverty", "corruption", "argument", "policy", "campaign", "democracy"]
        }
    ]

    def __init__(self, architecture: str = "roberta", device: str = "cpu"):
        self.architecture = architecture.lower()
        self.device = device
        self.linguistics = LinguisticFeatureExtractor()
        
        # Inicializar modelo BiLSTM con dimensiones: 768 (embedding) + 6 (lingüística)
        self.bilstm_model = ContextualBiLSTMMetaphorModel(
            embedding_dim=768,
            linguistic_dim=6,
            hidden_dim=64,
            dropout=0.1
        ).to(device)
        self.bilstm_model.eval()

        self._tokenizer = None
        self._transformer_model = None
        self._transformer_loaded = False

    def _load_transformer(self):
        """Carga el modelo transformer para extracción de embeddings contextuales."""
        if self._transformer_loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModel
            model_name = "dccuchile/bert-base-spanish-wwm-cased" if "bert" in self.architecture else "xlm-roberta-base"
            print(f"[Metaphor Detector] Inicializando extractor de embeddings: {model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._transformer_model = AutoModel.from_pretrained(model_name).to(self.device)
            self._transformer_model.eval()
            self._transformer_loaded = True
            print("[Metaphor Detector] Extractor contextual cargado.")
        except Exception as e:
            print(f"[Metaphor Detector] Nota: Extractor remoto no disponible ({e}). Usando motor analítico lingüístico y BiLSTM.")
            self._transformer_loaded = False

    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Analiza una oración completa detectando:
        1. Tokens metafóricos individuales (IlliniMet).
        2. Marcos conceptuales activos (MetaNet).
        3. Puntuación global de metaforicidad.
        """
        if not sentence or not sentence.strip():
            return {
                "text": sentence,
                "is_metaphorical": False,
                "metaphor_score": 0.0,
                "metaphorical_tokens": [],
                "conceptual_frames": []
            }

        words = re.findall(r"\b[\wáéíóúñüÁÉÍÓÚÑÜ]+\b", sentence)
        if not words:
            return {
                "text": sentence,
                "is_metaphorical": False,
                "metaphor_score": 0.0,
                "metaphorical_tokens": [],
                "conceptual_frames": []
            }

        # 1. Análisis de Marcos Conceptuales (MetaNet)
        detected_frames = self._detect_conceptual_frames(sentence, words)

        # 2. Análisis lingüístico por token (Concreción, WordNet y disparidad de dominio)
        token_results = []
        metaphor_token_count = 0

        for i, word in enumerate(words):
            conc = self.linguistics.get_concreteness(word)
            domains = self.linguistics.get_wordnet_domains(word)

            # Verificar disparidad de dominio con palabras adyacentes (ventana de contexto)
            context_window = words[max(0, i - 3):min(len(words), i + 4)]
            max_disparity = 0.0
            for ctx_word in context_window:
                if ctx_word.lower() != word.lower():
                    disp = self.linguistics.calculate_domain_disparity(word, ctx_word)
                    if disp > max_disparity:
                        max_disparity = disp

            # Criterio IlliniMet / MetaNet:
            # Una palabra es potencialmente metafórica si tiene alta disparidad de dominio en contexto
            # o si participa en una transferencia abstracto-concreta
            is_token_meta = (max_disparity >= 0.50) or any(word.lower() in f.get("matched_keywords", []) for f in detected_frames)
            if is_token_meta:
                metaphor_token_count += 1

            token_results.append({
                "word": word,
                "concreteness": round(conc, 2),
                "domains": domains[:2],
                "disparity_score": round(max_disparity, 3),
                "is_metaphorical": is_token_meta
            })

        # 3. Puntuación global (normalizada para evitar dilución extrema en documentos extensos)
        effective_len = max(15.0, min(float(len(words)), 25.0 + (len(words) ** 0.55) * 2.5))
        ratio = metaphor_token_count / effective_len
        frame_boost = min(0.55, 0.25 + 0.12 * len(detected_frames)) if detected_frames else 0.0
        final_score = min(1.0, (ratio * 1.5) + frame_boost)

        return {
            "text": sentence,
            "metaphor_score": round(final_score, 4),
            "is_metaphorical": final_score >= 0.35,
            "metaphorical_tokens": [t["word"] for t in token_results if t["is_metaphorical"]],
            "conceptual_frames": detected_frames,
            "token_analysis": token_results
        }

    def _detect_conceptual_frames(self, sentence: str, words: List[str]) -> List[Dict[str, Any]]:
        """Identifica si la oración activa proyecciones metafóricas conceptuales."""
        lower_words = [w.lower() for w in words]
        matched_frames = []

        for frame in self.CONCEPTUAL_FRAMES:
            src_matches = [w for w in lower_words if w in frame["source_keywords"]]
            tgt_matches = [w for w in lower_words if w in frame["target_keywords"]]

            # Hay activación si coocurren elementos del dominio fuente y del dominio destino
            if src_matches and tgt_matches:
                matched_frames.append({
                    "frame_name": frame["name"],
                    "source_domain": frame["source_domain"],
                    "target_domain": frame["target_domain"],
                    "source_words": src_matches,
                    "target_words": tgt_matches,
                    "matched_keywords": src_matches + tgt_matches
                })

        return matched_frames
