package com.nlp.pipeline.model;

import java.util.HashMap;
import java.util.Map;

/**
 * Representa el perfil emocional multidimensional basado en el Léxico de Emociones NRC (NRC EmoLex)
 * y el NRC Affective Intensity Lexicon, evitando escalas escalares simples de polaridad.
 * Modela las 8 emociones básicas de Plutchik:
 * - Ira (Anger)
 * - Asco (Disgust)
 * - Miedo (Fear)
 * - Tristeza (Sadness)
 * - Sorpresa (Surprise)
 * - Anticipación (Anticipation)
 * - Alegría (Joy)
 * - Confianza (Trust)
 */
public class EmotionProfile {

    public double anger;         // Ira
    public double disgust;       // Asco
    public double fear;          // Miedo
    public double sadness;       // Tristeza
    public double surprise;      // Sorpresa
    public double anticipation;  // Anticipación
    public double joy;           // Alegría
    public double trust;         // Confianza

    public double averageIntensity; // Intensidad afectiva media (0.0 a 1.0)
    public int emotionalWordCount;   // Cantidad de palabras con carga emocional
    public int totalWordsAnalyzed;   // Total de palabras evaluadas
    public String predominantEmotion; // Emoción con mayor peso

    // Distribución por etiquetas gramaticales (POS tags)
    public Map<String, Double> posEmotionWeights = new HashMap<>();

    public EmotionProfile() {
        this.predominantEmotion = "neutral";
    }

    /**
     * Retorna el vector de características de 8 dimensiones normalizado (suma = 1.0 si hay emociones).
     * Ideal para alimentar modelos de clasificación o clustering.
     */
    public double[] getNormalizedEmotionVector() {
        double total = anger + disgust + fear + sadness + surprise + anticipation + joy + trust;
        if (total == 0.0) {
            return new double[8];
        }
        return new double[]{
            anger / total,
            disgust / total,
            fear / total,
            sadness / total,
            surprise / total,
            anticipation / total,
            joy / total,
            trust / total
        };
    }

    /**
     * Retorna el vector de características ampliado (8 emociones normalizadas + 1 dimensión de intensidad afectiva).
     */
    public double[] getFeatureVectorWithIntensity() {
        double[] norm = getNormalizedEmotionVector();
        double[] vector = new double[9];
        System.arraycopy(norm, 0, vector, 0, 8);
        vector[8] = averageIntensity;
        return vector;
    }

    public Map<String, Double> toEmotionMap() {
        Map<String, Double> map = new HashMap<>();
        map.put("ira", anger);
        map.put("asco", disgust);
        map.put("miedo", fear);
        map.put("tristeza", sadness);
        map.put("sorpresa", surprise);
        map.put("anticipacion", anticipation);
        map.put("alegria", joy);
        map.put("confianza", trust);
        return map;
    }

    @Override
    public String toString() {
        return String.format(
            "Emoción dominante: %s | Intensidad: %.2f | Palabras afectivas: %d/%d\n" +
            "  [Ira: %.2f, Asco: %.2f, Miedo: %.2f, Tristeza: %.2f, " +
            "Sorpresa: %.2f, Anticipación: %.2f, Alegría: %.2f, Confianza: %.2f]",
            predominantEmotion, averageIntensity, emotionalWordCount, totalWordsAnalyzed,
            anger, disgust, fear, sadness, surprise, anticipation, joy, trust
        );
    }
}
