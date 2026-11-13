"""
Módulo de Detección de Metáforas:
- Red Neuronal Recurrente Bidireccional (BiLSTM) alimentada por embeddings contextuales (BERT, RoBERTa, XLNet).
- Características lingüísticas de Abstracción y Concreción.
- Conjuntos de sinónimos y dominios semánticos desambiguados extraídos de WordNet/VerbNet.
- Principios de IlliniMet (RoBERTa + WordNet a nivel de palabra) y MetaNet (Teoría de Metáfora Conceptual y Semántica de Marcos).
"""

from .linguistic_features import LinguisticFeatureExtractor
from .bilstm_model import ContextualBiLSTMMetaphorModel
from .metaphor_detector import MetaphorDetector

__all__ = [
    "LinguisticFeatureExtractor",
    "ContextualBiLSTMMetaphorModel",
    "MetaphorDetector"
]
