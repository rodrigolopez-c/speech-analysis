"""
Detección de Sarcasmo usando Hugging Face Transformers:
Soporte para arquitecturas contextuales de última generación:
- BERT (BETO / Multilingual BERT)
- RoBERTa (XLM-RoBERTa / RoBERTa-base-irony)
- XLNet
Permite analizar oraciones considerando su contexto conversacional precedente.
"""

import os
from typing import List, Dict, Any, Optional, Tuple


class TransformerSarcasmDetector:
    """
    Detector de sarcasmo basado en modelos Transformer preentrenados de Hugging Face.
    Analiza tanto oraciones individuales como pares (contexto, oración objetivo).
    """

    SUPPORTED_ARCHITECTURES = {
        "roberta": "cardiffnlp/twitter-roberta-base-irony",
        "bert": "dccuchile/bert-base-spanish-wwm-cased",
        "xlnet": "xlnet-base-cased"
    }

    def __init__(self, architecture: str = "roberta", model_name_or_path: Optional[str] = None, device: str = "cpu"):
        self.architecture = architecture.lower()
        self.device = device
        self.model_name = model_name_or_path or self.SUPPORTED_ARCHITECTURES.get(self.architecture, "roberta-base")
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._is_initialized = False

    def initialize(self):
        """Inicializa el modelo Transformer y tokenizador de forma perezosa (lazy)."""
        if self._is_initialized:
            return

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            print(f"[Transformer Sarcasm] Cargando modelo: {self.model_name} (Arquitectura: {self.architecture})...")
            # Cargar pipeline de clasificación de secuencia
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1 if self.device == "cpu" else 0,
                top_k=None
            )
            self._is_initialized = True
            print("[Transformer Sarcasm] Modelo inicializado exitosamente.")
        except Exception as e:
            print(f"[Transformer Sarcasm] Nota: no fue posible descargar/cargar pesos de red remota ({e}).")
            print("[Transformer Sarcasm] Activando modo de análisis de contexto e incongruencia contextual.")
            self._is_initialized = False

    def predict_sentence(self, sentence: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analiza una oración considerando su contexto conversacional.
        Retorna la probabilidad de sarcasmo (0.0 a 1.0) y etiqueta predicha.
        """
        if not self._is_initialized:
            self.initialize()

        # Construir entrada considerando contexto previo
        if context and context.strip():
            formatted_input = f"Contexto: {context.strip()} | Expresión: {sentence.strip()}"
        else:
            formatted_input = sentence.strip()

        if self.pipeline is not None:
            try:
                outputs = self.pipeline(formatted_input)
                # Formato outputs: [[{'label': 'LABEL_1', 'score': 0.85}, ...]]
                results = outputs[0] if isinstance(outputs, list) and isinstance(outputs[0], list) else outputs
                
                sarcasm_score = 0.0
                for item in results:
                    label = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.0))
                    if "irony" in label or "sarcasm" in label or label == "label_1" or "1" in label:
                        sarcasm_score = score
                        break
                
                return {
                    "text": sentence,
                    "context": context,
                    "sarcasm_probability": round(sarcasm_score, 4),
                    "is_sarcastic": sarcasm_score >= 0.5,
                    "architecture": self.architecture,
                    "model": self.model_name
                }
            except Exception as e:
                print(f"[Transformer Sarcasm] Fallback en inferencia: {e}")

        # Fallback analítico basado en contraste de contexto si el modelo remoto no está disponible
        return self._contextual_fallback_analysis(sentence, context)

    def predict_batch(self, pairs: List[Tuple[str, Optional[str]]]) -> List[Dict[str, Any]]:
        """Procesa una lista de tuplas (oración, contexto_opcional)."""
        return [self.predict_sentence(sentence, ctx) for sentence, ctx in pairs]

    def _contextual_fallback_analysis(self, sentence: str, context: Optional[str]) -> Dict[str, Any]:
        """
        Algoritmo contextual heurístico de respaldo:
        Compara la valencia emocional de la oración con el tono de la discusión o contexto.
        """
        from .semantic_incongruity import SemanticIncongruityDetector
        detector = SemanticIncongruityDetector()
        score = detector.calculate_incongruity(sentence, context)
        return {
            "text": sentence,
            "context": context,
            "sarcasm_probability": round(score, 4),
            "is_sarcastic": score >= 0.5,
            "architecture": f"{self.architecture} (context-incongruity fallback)",
            "model": "rule-context-engine"
        }
