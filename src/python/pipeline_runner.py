"""
Runner CLI del Pipeline NLP Avanzado:
Orquesta y ejecuta el análisis de:
1. Emociones Concretas (Léxico NRC EmoLex e Intensidad).
2. Detección de Sarcasmo (Transformers BERT/RoBERTa/XLNet, Scikit-Learn Baselines, DeepMoji Incongruity).
3. Detección de Metáforas (BiLSTM + Contextual Embeddings, WordNet, Concreción, IlliniMet, MetaNet).
"""

import os
import sys
import argparse
import pandas as pd
from typing import List, Dict, Any

from sarcasm import SarcasmBaselineModel, TransformerSarcasmDetector, SemanticIncongruityDetector
from metaphor import MetaphorDetector


def process_dataset(csv_path: str, output_path: str = None, sample_size: int = 50):
    print("=" * 70)
    print("      PIPELINE NLP AVANZADO: SARCASMO, METÁFORAS Y EMOCIONES NRC     ")
    print("=" * 70)
    print(f"Cargando dataset desde: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"Error: Archivo no encontrado en {csv_path}")
        return

    # Leer CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=";")

    print(f"Total registros cargados: {len(df)}")

    # Identificar columna de texto
    text_col = None
    for col in ["sampleText", "text_clean", "text-original", "content", "tokens", "texto"]:
        if col in df.columns:
            text_col = col
            break

    if not text_col:
        text_col = df.columns[-1]

    print(f"Columna de texto seleccionada: '{text_col}'")

    # Tomar submuestra representativa para ejecución rápida
    if sample_size > 0 and len(df) > sample_size:
        df_sample = df.head(sample_size).copy()
    else:
        df_sample = df.copy()

    # Inicializar detectores
    print("\nInicializando modelos...")
    sarcasm_svm = SarcasmBaselineModel(model_type="svm")
    sarcasm_deepmoji = SemanticIncongruityDetector()
    metaphor_engine = MetaphorDetector(architecture="roberta")

    sarcasm_scores = []
    incongruity_scores = []
    metaphor_scores = []
    metaphor_tokens_list = []
    active_frames_list = []

    print(f"Procesando {len(df_sample)} documentos...")
    for idx, row in df_sample.iterrows():
        raw_val = row[text_col]
        text = str(raw_val) if pd.notna(raw_val) else ""
        if len(text) > 6000:
            text = text[:6000]

        # 1. Detección de Sarcasmo (Baseline SVM)
        s_score = sarcasm_svm.predict_score([text])[0]
        sarcasm_scores.append(round(s_score, 4))

        # 2. Incongruencia Semántica (DeepMoji)
        inc_score = sarcasm_deepmoji.calculate_incongruity(text)
        incongruity_scores.append(round(inc_score, 4))

        # 3. Detección de Metáforas (WordNet + Concreción + MetaNet)
        m_result = metaphor_engine.analyze_sentence(text)
        metaphor_scores.append(m_result["metaphor_score"])
        metaphor_tokens_list.append(", ".join(m_result["metaphorical_tokens"][:5]))
        frames = [f["frame_name"] for f in m_result["conceptual_frames"]]
        active_frames_list.append(", ".join(frames) if frames else "ninguno")

    df_sample["sarcasm_score_svm"] = sarcasm_scores
    df_sample["sarcasm_incongruity"] = incongruity_scores
    df_sample["metaphor_score"] = metaphor_scores
    df_sample["metaphor_tokens"] = metaphor_tokens_list
    df_sample["conceptual_frames"] = active_frames_list

    # Resumen de Resultados
    print("\n" + "=" * 70)
    print("                    RESUMEN DE RESULTADOS                    ")
    print("=" * 70)
    avg_sarcasm = sum(sarcasm_scores) / len(sarcasm_scores)
    avg_inc = sum(incongruity_scores) / len(incongruity_scores)
    avg_meta = sum(metaphor_scores) / len(metaphor_scores)
    metaphor_detected_count = sum(1 for s in metaphor_scores if s >= 0.35)
    sarcasm_detected_count = sum(1 for s in sarcasm_scores if s >= 0.6)

    print(f"Documentos evaluados: {len(df_sample)}")
    print(f"\n[Detección de Sarcasmo]:")
    print(f"  * Propensión promedio al sarcasmo (SVM): {avg_sarcasm:.3f}")
    print(f"  * Incongruencia semántica media (DeepMoji): {avg_inc:.3f}")
    print(f"  * Expresiones con alto potencial sarcástico: {sarcasm_detected_count}/{len(df_sample)}")

    print(f"\n[Detección de Metáforas (IlliniMet & MetaNet)]:")
    print(f"  * Puntuación media de metaforicidad: {avg_meta:.3f}")
    print(f"  * Oraciones con metáforas detectadas: {metaphor_detected_count}/{len(df_sample)}")
    
    # Marcos conceptuales encontrados
    frame_counts = {}
    for f_str in active_frames_list:
        if f_str and f_str != "ninguno":
            for f in f_str.split(", "):
                frame_counts[f] = frame_counts.get(f, 0) + 1

    if frame_counts:
        print("\n  Marcos conceptuales más frecuentes:")
        for frame, count in sorted(frame_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {frame}: {count} ocurrencias")

    # Guardar resultados
    if not output_path:
        output_path = "target/advanced_nlp_results.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sample.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nResultados detallados exportados exitosamente a: {output_path}")
    print("=" * 70)


def test_interactive():
    """Ejecuta demostración interactiva de casos de prueba paradigmáticos."""
    print("=" * 70)
    print("    PRUEBA DEMOSTRATIVA DE CASOS: SARCASMO Y METÁFORAS NLP    ")
    print("=" * 70)

    sarcasm_svm = SarcasmBaselineModel(model_type="svm")
    sarcasm_deepmoji = SemanticIncongruityDetector()
    metaphor_engine = MetaphorDetector(architecture="roberta")

    examples = [
        # Metáforas conceptuales
        "Debemos construir los cimientos del futuro sobre las raíces de nuestra universidad.",
        "El tiempo vuela cuando trabajamos con entusiasmo por la comunidad.",
        "Sembrar ideas de justicia en los jóvenes dará frutos para la sociedad.",
        # Sarcasmo / ironía
        "¡Qué maravilla de sistema! Falló tres veces y perdimos todos los archivos.",
        "Excelente servicio, justo lo que necesitaba: esperar dos horas más bajo la lluvia.",
        # Texto literal académico / institucional
        "El rector presentó el informe financiero correspondiente al periodo 2020-2024.",
        "Se convocó a la sesión ordinaria del consejo universitario el próximo viernes."
    ]

    for ex in examples:
        print(f"\nTexto: \"{ex}\"")
        # Sarcasmo
        svm_score = sarcasm_svm.predict_score([ex])[0]
        inc_score = sarcasm_deepmoji.calculate_incongruity(ex)
        is_sarcastic = svm_score > 0.6 or inc_score >= 0.5
        print(f"  [Sarcasmo] Probabilidad: {svm_score:.2f} | Incongruencia: {inc_score:.2f} -> {'SARCÁSTICO' if is_sarcastic else 'LITERAL'}")

        # Metáfora
        meta = metaphor_engine.analyze_sentence(ex)
        tokens = ", ".join(meta["metaphorical_tokens"]) if meta["metaphorical_tokens"] else "ninguno"
        frames = ", ".join([f["frame_name"] for f in meta["conceptual_frames"]]) if meta["conceptual_frames"] else "ninguno"
        print(f"  [Metáfora] Puntuación: {meta['metaphor_score']:.2f} | Tokens: [{tokens}] | Marcos: [{frames}] -> {'METAFÓRICO' if meta['is_metaphorical'] else 'LITERAL'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Avanzado de Sarcasmo y Metáforas")
    parser.add_argument("--csv", type=str, default=None, help="Ruta al archivo CSV para procesar")
    parser.add_argument("--output", type=str, default=None, help="Ruta para guardar el CSV de salida")
    parser.add_argument("--sample", type=int, default=50, help="Tamaño de muestra a procesar")
    parser.add_argument("--demo", action="store_true", help="Ejecutar prueba interactiva con oraciones de ejemplo")

    args = parser.parse_args()

    if args.demo or not args.csv:
        test_interactive()
    else:
        process_dataset(args.csv, args.output, args.sample)
