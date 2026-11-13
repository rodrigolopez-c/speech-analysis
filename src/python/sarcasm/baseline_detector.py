"""
Modelos de Línea Base para Detección de Sarcasmo usando Scikit-Learn.
Implementa Support Vector Machines (SVM) y Regresión Logística
con características estadísticas, léxicas y vectores de emociones.
"""

import numpy as np
import re
from typing import List, Dict, Any, Tuple
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score


class SarcasmStatisticalFeatures(BaseEstimator, TransformerMixin):
    """
    Extractor de características estadísticas e incongruencias locales:
    - Puntuación enfática (!, ?, !?, ...)
    - Razón de mayúsculas (grito / énfasis)
    - Repetición de caracteres (alargamiento léxico: 'claaro', 'siii')
    - Disparidad de polaridad (palabras positivas en contraste con puntuación o contexto negativo)
    """

    def fit(self, x, y=None):
        return self

    def transform(self, texts: List[str]):
        features = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text or "")
            
            length = max(len(text), 1)
            exclamations = text.count("!")
            questions = text.count("?")
            quotes = text.count('"') + text.count("'")
            dots = text.count("...")
            
            # Ratio de mayúsculas
            upper_chars = sum(1 for c in text if c.isupper())
            caps_ratio = upper_chars / length
            
            # Detección de alargamiento de caracteres (e.g., 'buueenoo')
            elongation = len(re.findall(r"(\w)\1{2,}", text))
            
            # Palabras positivas y negativas frecuentes en sarcasmo irónico
            pos_markers = ["claro", "seguro", "maravilla", "genial", "excelente", "perfecto", "gran", "obvio", "sure", "great", "nice"]
            neg_markers = ["pero", "jamas", "nunca", "terrible", "peor", "horrible", "nada", "but", "never", "worst"]
            
            lower_text = text.lower()
            pos_count = sum(1 for m in pos_markers if m in lower_text)
            neg_count = sum(1 for m in neg_markers if m in lower_text)
            
            # Incongruencia: combinación simultánea de marcadores positivos y negativos
            incongruity = 1.0 if (pos_count > 0 and neg_count > 0) else 0.0

            row = [
                exclamations / length,
                questions / length,
                quotes / length,
                dots / length,
                caps_ratio,
                elongation,
                pos_count,
                neg_count,
                incongruity
            ]
            features.append(row)
            
        return np.array(features)


class SarcasmBaselineModel:
    """
    Pipeline de Scikit-Learn para clasificación de sarcasmo usando:
    - SVM (Support Vector Machines)
    - Regresión Logística
    """

    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type.lower()
        self.pipeline = self._build_pipeline()
        self.is_trained = False

    def _build_pipeline(self) -> Pipeline:
        feature_extractor = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
            ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2000)),
            ("stats", SarcasmStatisticalFeatures())
        ])

        if self.model_type == "logistic_regression" or self.model_type == "lr":
            classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        else:
            classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=42)

        return Pipeline([
            ("features", feature_extractor),
            ("clf", classifier)
        ])

    def train(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Entrena el modelo de línea base y retorna métricas de ajuste."""
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        preds = self.pipeline.predict(texts)
        return {
            "model_type": self.model_type,
            "accuracy": accuracy_score(labels, preds),
            "f1_score": f1_score(labels, preds, average="weighted", zero_division=0),
            "report": classification_report(labels, preds, zero_division=0)
        }

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predice etiquetas de sarcasmo (1 = sarcástico, 0 = literal)."""
        if not self.is_trained:
            # Si no ha sido entrenado externamente, inicializa con un baseline pre-calibrado
            self._fit_default_baseline()
        return self.pipeline.predict(texts)

    def predict_score(self, texts: List[str]) -> List[float]:
        """Retorna puntuaciones continuas de propensión al sarcasmo (0.0 a 1.0)."""
        if not self.is_trained:
            self._fit_default_baseline()

        clf = self.pipeline.named_steps["clf"]
        transformed = self.pipeline.named_steps["features"].transform(texts)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(transformed)
            return [float(p[1]) for p in probs]
        elif hasattr(clf, "decision_function"):
            df = clf.decision_function(transformed)
            # Sigmoide para normalizar decision_function a rango [0, 1]
            return [float(1.0 / (1.0 + np.exp(-val))) for val in df]
        else:
            preds = clf.predict(transformed)
            return [float(p) for p in preds]

    def _fit_default_baseline(self):
        """Entrena con un conjunto curado de calibración para casos donde no se proporcione dataset etiquetado."""
        sample_texts = [
            "Excelente trabajo, arruinaron todo como siempre.",
            "¡Claro que sí! Justo lo que necesitaba, más retrasos y problemas.",
            "Me encanta esperar tres horas en una fila sin sentido.",
            "Qué maravilla de sistema, falla cada vez que intento guardar algo.",
            "Brillante idea, ahora tenemos el doble de trabajo y la mitad del presupuesto.",
            "El rector presentó el informe anual con resultados positivos en docencia.",
            "La reunión del consejo universitario se llevará a cabo el próximo lunes.",
            "Agradecemos profundamente el compromiso de toda la comunidad universitaria.",
            "Se implementaron nuevos programas de investigación académica.",
            "Los estudiantes participaron activamente en los talleres de innovación social."
        ]
        sample_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        self.train(sample_texts, sample_labels)
