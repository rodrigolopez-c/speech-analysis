"""
Módulo de Detección de Sarcasmo:
- Modelos de Línea Base con Scikit-Learn (SVM, Regresión Logística).
- Modelos de Lenguaje Contextuales con Hugging Face Transformers (BERT, RoBERTa, XLNet).
- Detección de Incongruencias Semánticas y Estados Emocionales (Inspirado en DeepMoji).
"""

from .baseline_detector import SarcasmBaselineModel
from .transformer_detector import TransformerSarcasmDetector
from .semantic_incongruity import SemanticIncongruityDetector

__all__ = [
    "SarcasmBaselineModel",
    "TransformerSarcasmDetector",
    "SemanticIncongruityDetector"
]
