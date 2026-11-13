package com.nlp.pipeline.nlp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nlp.pipeline.model.EmotionProfile;

/**
 * Servicio para análisis de emociones basado en:
 * 1. Léxico de Emociones NRC (NRC EmoLex): 8 emociones básicas de Plutchik.
 * 2. NRC Affective Intensity Lexicon: mide la fuerza/intensidad del afecto.
 * 3. Integración con StanfordNLP: ponderación por etiquetas gramaticales (POS tags).
 */
public class NRCEmotionService {

    public static class NRCEntry {
        public final int anger;
        public final int disgust;
        public final int fear;
        public final int sadness;
        public final int surprise;
        public final int anticipation;
        public final int joy;
        public final int trust;
        public final double intensity;
        public final int polarity;

        public NRCEntry(int anger, int disgust, int fear, int sadness, int surprise,
                        int anticipation, int joy, int trust, double intensity, int polarity) {
            this.anger = anger;
            this.disgust = disgust;
            this.fear = fear;
            this.sadness = sadness;
            this.surprise = surprise;
            this.anticipation = anticipation;
            this.joy = joy;
            this.trust = trust;
            this.intensity = intensity;
            this.polarity = polarity;
        }
    }

    private final Map<String, NRCEntry> lexicon = new HashMap<>();

    public NRCEmotionService() {
        loadLexicon();
    }

    private void loadLexicon() {
        String resourcePath = "lexicons/nrc_emolex.csv";
        InputStream is = getClass().getClassLoader().getResourceAsStream(resourcePath);

        if (is == null) {
            File fallbackFile = new File("src/main/resources/lexicons/nrc_emolex.csv");
            if (fallbackFile.exists()) {
                try {
                    is = new FileInputStream(fallbackFile);
                } catch (Exception ignored) {
                }
            }
        }

        if (is == null) {
            System.err.println("ADVERTENCIA: No se encontró el léxico NRC en " + resourcePath);
            return;
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            boolean isHeader = true;
            int loaded = 0;

            while ((line = reader.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length >= 10) {
                    String word = parts[0].trim().toLowerCase();
                    int anger = Integer.parseInt(parts[1].trim());
                    int disgust = Integer.parseInt(parts[2].trim());
                    int fear = Integer.parseInt(parts[3].trim());
                    int sadness = Integer.parseInt(parts[4].trim());
                    int surprise = Integer.parseInt(parts[5].trim());
                    int anticipation = Integer.parseInt(parts[6].trim());
                    int joy = Integer.parseInt(parts[7].trim());
                    int trust = Integer.parseInt(parts[8].trim());
                    double intensity = Double.parseDouble(parts[9].trim());
                    int polarity = parts.length > 10 ? Integer.parseInt(parts[10].trim()) : 0;

                    lexicon.put(word, new NRCEntry(anger, disgust, fear, sadness, surprise,
                            anticipation, joy, trust, intensity, polarity));
                    loaded++;
                }
            }
            System.out.println("Léxico NRC EmoLex cargado exitosamente: " + loaded + " entradas (Español/Inglés con Intensidad).");
        } catch (Exception e) {
            System.err.println("Error al cargar léxico NRC: " + e.getMessage());
        }
    }

    /**
     * Pondera el impacto emocional de una palabra según su etiqueta gramatical (POS tag de StanfordNLP).
     * Los adjetivos y adverbios suelen cualificar intensamente el estado emocional, seguidos por verbos y sustantivos.
     */
    public double getPOSWeight(String posTag) {
        if (posTag == null || posTag.isEmpty()) {
            return 1.0;
        }
        String upper = posTag.toUpperCase();
        if (upper.startsWith("JJ") || upper.equals("ADJ")) {
            return 1.5; // Adjetivo: máxima carga descriptiva emocional
        } else if (upper.startsWith("RB") || upper.equals("ADV")) {
            return 1.3; // Adverbio: modulador de intensidad
        } else if (upper.startsWith("VB") || upper.equals("VERB")) {
            return 1.2; // Verbo: acción / estado emocional
        } else if (upper.startsWith("NN") || upper.equals("NOUN")) {
            return 1.0; // Sustantivo: referencia afectiva base
        }
        return 0.8; // Otras categorías gramaticales
    }

    /**
     * Construye un EmotionProfile integrando tokens, POS tags de StanfordNLP e intensidad afectiva.
     */
    public EmotionProfile analyzeTokensWithPOS(List<String> tokens, List<String> posTags) {
        EmotionProfile profile = new EmotionProfile();
        if (tokens == null || tokens.isEmpty()) {
            return profile;
        }

        profile.totalWordsAnalyzed = tokens.size();
        double sumIntensity = 0.0;
        int emotionalWords = 0;

        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i).toLowerCase().trim();
            // Limpieza básica de puntuación
            token = token.replaceAll("^[^a-záéíóúñü]+|[^a-záéíóúñü]+$", "");
            if (token.isEmpty()) continue;

            NRCEntry entry = lexicon.get(token);
            if (entry != null) {
                String pos = (posTags != null && i < posTags.size()) ? posTags.get(i) : "";
                double posWeight = getPOSWeight(pos);

                // Multiplicamos por la intensidad afectiva de NRC y por el peso del POS
                double factor = posWeight * entry.intensity;

                if (entry.anger > 0) profile.anger += factor;
                if (entry.disgust > 0) profile.disgust += factor;
                if (entry.fear > 0) profile.fear += factor;
                if (entry.sadness > 0) profile.sadness += factor;
                if (entry.surprise > 0) profile.surprise += factor;
                if (entry.anticipation > 0) profile.anticipation += factor;
                if (entry.joy > 0) profile.joy += factor;
                if (entry.trust > 0) profile.trust += factor;

                boolean hasEmotion = (entry.anger + entry.disgust + entry.fear + entry.sadness +
                        entry.surprise + entry.anticipation + entry.joy + entry.trust) > 0;

                if (hasEmotion) {
                    emotionalWords++;
                    sumIntensity += entry.intensity;
                    String posCat = (pos != null && pos.length() >= 2) ? pos.substring(0, 2) : "OTHER";
                    profile.posEmotionWeights.put(posCat, profile.posEmotionWeights.getOrDefault(posCat, 0.0) + factor);
                }
            }
        }

        profile.emotionalWordCount = emotionalWords;
        profile.averageIntensity = (emotionalWords > 0) ? (sumIntensity / emotionalWords) : 0.0;
        profile.predominantEmotion = computePredominantEmotion(profile);

        return profile;
    }

    /**
     * Determina la emoción predominante basada en las 8 dimensiones.
     */
    private String computePredominantEmotion(EmotionProfile p) {
        double max = 0.0;
        String predominant = "neutral";

        if (p.trust > max) { max = p.trust; predominant = "confianza"; }
        if (p.joy > max) { max = p.joy; predominant = "alegria"; }
        if (p.anticipation > max) { max = p.anticipation; predominant = "anticipacion"; }
        if (p.surprise > max) { max = p.surprise; predominant = "sorpresa"; }
        if (p.fear > max) { max = p.fear; predominant = "miedo"; }
        if (p.sadness > max) { max = p.sadness; predominant = "tristeza"; }
        if (p.anger > max) { max = p.anger; predominant = "ira"; }
        if (p.disgust > max) { max = p.disgust; predominant = "asco"; }

        return max > 0.0 ? predominant : "neutral";
    }

    /**
     * Mapeo auxiliar de compatibilidad: convierte el perfil de emoción en un puntaje continuo o 1-3.
     */
    public double toPolarityScore(EmotionProfile profile) {
        double positiveWeights = profile.joy + profile.trust + (profile.anticipation * 0.5);
        double negativeWeights = profile.anger + profile.disgust + profile.fear + profile.sadness;
        double diff = positiveWeights - negativeWeights;

        if (diff > 0.5) return 3.0; // Predominio positivo
        if (diff < -0.5) return 1.0; // Predominio negativo
        return 2.0; // Neutral / equilibrado
    }
}
