"""
Detector de Incongruencias Semánticas y Emocionales Locales (Inspirado en DeepMoji).
Captura estados afectivos contrastantes en el texto:
- Discrepancia entre emoción aparente superficial y valencia negativa subyacente.
- Tensión emocional local entre cláusulas consecutivas.
- Incongruencia conversacional entre el contexto y la respuesta.
"""

import re
from typing import Optional, Dict, Any, List


class SemanticIncongruityDetector:
    """
    Modela estados emocionales e incongruencias semánticas en el texto:
    Principio fundamental del sarcasmo según la teoría de DeepMoji y lingüística computacional:
    la presencia simultánea de alta valencia positiva verbal y contexto/eventos fuertemente negativos.
    """

    def __init__(self):
        # Términos típicamente de valencia superficial altamente positiva
        self.superficial_positive = {
            "maravilloso", "maravilla", "excelente", "fantástico", "increíble", "genial",
            "perfecto", "brillante", "espléndido", "gran", "fascinante", "encantado",
            "great", "amazing", "wonderful", "perfect", "brilliant", "lovely"
        }

        # Términos / eventos intrínsecamente negativos o frustrantes
        self.negative_events = {
            "arruinar", "arruinaron", "fallar", "falla", "error", "tarde", "retraso", "esperar",
            "fila", "problema", "peor", "horrible", "desastre", "caos", "dolor", "tráfico",
            "basura", "lluvia", "pesadilla", "ruined", "failed", "late", "delay", "worst", "mess"
        }

        # Marcadores explícitos de ironía / contraste
        self.contrast_markers = {
            "pero", "sin embargo", "aunque", "no obstante", "claro que", "por supuesto que",
            "obvio", "seguro que", "but", "however", "obviously", "surely"
        }

    def calculate_incongruity(self, text: str, context: Optional[str] = None) -> float:
        """
        Calcula un puntaje continuo de incongruencia emocional y semántica entre 0.0 y 1.0.
        Valores cercanos a 1.0 denotan alta probabilidad de incongruencia sarcástica.
        """
        if not text:
            return 0.0

        lower_text = text.lower()
        words = re.findall(r"\b\w+\b", lower_text)

        pos_count = sum(1 for w in words if w in self.superficial_positive)
        neg_count = sum(1 for w in words if w in self.negative_events)
        contrast_count = sum(1 for m in self.contrast_markers if m in lower_text)

        # 1. Incongruencia intra-oracional: coocurrencia de positivo y negativo
        intra_incongruity = 0.0
        if pos_count > 0 and neg_count > 0:
            intra_incongruity = min(1.0, 0.5 + 0.2 * (pos_count + neg_count))
        elif pos_count > 0 and ("!" in text or "¿" in text or "?" in text):
            # Positivo exagerado con puntuación enfática
            intra_incongruity = 0.35

        # 2. Incongruencia contextual (context vs target)
        context_incongruity = 0.0
        if context and context.strip():
            ctx_lower = context.lower()
            ctx_words = re.findall(r"\b\w+\b", ctx_lower)
            ctx_neg = sum(1 for w in ctx_words if w in self.negative_events)
            
            # Si el contexto es negativo y la respuesta es superficialmente positiva: fuerte sarcasmo
            if ctx_neg > 0 and pos_count > 0:
                context_incongruity = min(1.0, 0.6 + 0.2 * ctx_neg)

        # 3. Presencia de marcadores irónicos de refuerzo
        marker_weight = min(0.3, contrast_count * 0.15)

        total_score = max(intra_incongruity, context_incongruity) + marker_weight
        return float(min(1.0, max(0.0, total_score)))

    def analyze_emotional_state(self, text: str) -> Dict[str, Any]:
        """
        Retorna un desglose del estado emocional y la incongruencia semántica detectada.
        """
        score = self.calculate_incongruity(text)
        return {
            "text": text,
            "incongruity_score": round(score, 4),
            "is_incongruent": score >= 0.5,
            "incongruity_type": "polaridad_contrastante" if score >= 0.5 else "consistente"
        }
