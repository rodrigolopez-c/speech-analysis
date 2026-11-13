package com.nlp.pipeline.pipeline;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.model.EmotionProfile;

/**
 * Resumen de análisis de sentimientos y emociones concretas basado en el Léxico de Emociones NRC
 * (NRC EmoLex: Plutchik 8 emotions) y el NRC Affective Intensity Lexicon.
 */
public class SentimentSummary {

    public static void printSummary(List<DocumentResult> docs) {
        if (docs == null || docs.isEmpty()) {
            System.out.println("\n===== ANÁLISIS DE EMOCIONES NRC (EMOLEX) =====");
            System.out.println("No hay documentos para analizar.");
            return;
        }

        double totalAnger = 0, totalDisgust = 0, totalFear = 0, totalSadness = 0;
        double totalSurprise = 0, totalAnticipation = 0, totalJoy = 0, totalTrust = 0;
        double totalIntensity = 0;
        int validDocs = 0;

        Map<String, Integer> predominantCounts = new HashMap<>();
        Map<String, double[]> categoryEmotionSum = new HashMap<>();
        Map<String, Integer> categoryDocCount = new HashMap<>();

        for (DocumentResult doc : docs) {
            EmotionProfile p = doc.emotionProfile;
            if (p == null) continue;

            validDocs++;
            totalAnger += p.anger;
            totalDisgust += p.disgust;
            totalFear += p.fear;
            totalSadness += p.sadness;
            totalSurprise += p.surprise;
            totalAnticipation += p.anticipation;
            totalJoy += p.joy;
            totalTrust += p.trust;
            totalIntensity += p.averageIntensity;

            predominantCounts.put(p.predominantEmotion, predominantCounts.getOrDefault(p.predominantEmotion, 0) + 1);

            String cat = doc.category != null ? doc.category : "Desconocido";
            double[] catArr = categoryEmotionSum.computeIfAbsent(cat, k -> new double[8]);
            double[] norm = p.getNormalizedEmotionVector();
            for (int i = 0; i < 8; i++) catArr[i] += norm[i];
            categoryDocCount.put(cat, categoryDocCount.getOrDefault(cat, 0) + 1);
        }

        if (validDocs == 0) {
            System.out.println("\n===== ANÁLISIS DE EMOCIONES NRC (EMOLEX) =====");
            System.out.println("No se generaron perfiles de emociones.");
            return;
        }

        double grandTotal = totalAnger + totalDisgust + totalFear + totalSadness +
                totalSurprise + totalAnticipation + totalJoy + totalTrust;
        if (grandTotal == 0) grandTotal = 1.0;

        double avgIntensity = totalIntensity / validDocs;

        System.out.println("\n=======================================================");
        System.out.println("       ANÁLISIS DE EMOCIONES CONCRETAS - NRC EMOLEX    ");
        System.out.println("=======================================================");
        System.out.printf("Documentos analizados: %d | Intensidad Afectiva Media: %.3f%n%n", validDocs, avgIntensity);

        System.out.println("Distribución General de las 8 Emociones de Plutchik:");
        System.out.println("-------------------------------------------------------");
        System.out.printf("  * Confianza (Trust):        %7.2f (%5.1f%%)%n", totalTrust, (totalTrust / grandTotal) * 100);
        System.out.printf("  * Alegría (Joy):            %7.2f (%5.1f%%)%n", totalJoy, (totalJoy / grandTotal) * 100);
        System.out.printf("  * Anticipación (Anticip.):  %7.2f (%5.1f%%)%n", totalAnticipation, (totalAnticipation / grandTotal) * 100);
        System.out.printf("  * Sorpresa (Surprise):      %7.2f (%5.1f%%)%n", totalSurprise, (totalSurprise / grandTotal) * 100);
        System.out.printf("  * Miedo (Fear):             %7.2f (%5.1f%%)%n", totalFear, (totalFear / grandTotal) * 100);
        System.out.printf("  * Tristeza (Sadness):       %7.2f (%5.1f%%)%n", totalSadness, (totalSadness / grandTotal) * 100);
        System.out.printf("  * Ira (Anger):              %7.2f (%5.1f%%)%n", totalAnger, (totalAnger / grandTotal) * 100);
        System.out.printf("  * Asco (Disgust):           %7.2f (%5.1f%%)%n", totalDisgust, (totalDisgust / grandTotal) * 100);

        System.out.println("\nEmoción Predominante por Documento:");
        for (Map.Entry<String, Integer> entry : predominantCounts.entrySet()) {
            System.out.printf("  - %-14s : %3d documentos (%5.1f%%)%n",
                    entry.getKey().toUpperCase(), entry.getValue(), (entry.getValue() * 100.0 / validDocs));
        }

        if (categoryEmotionSum.size() > 1) {
            System.out.println("\nPerfil Emocional por Categoría / Autor:");
            System.out.println("-------------------------------------------------------");
            String[] emoNames = {"Ira", "Asco", "Miedo", "Trist", "Sorpr", "Antic", "Alegr", "Confi"};
            for (Map.Entry<String, double[]> entry : categoryEmotionSum.entrySet()) {
                String cat = entry.getKey();
                int count = categoryDocCount.get(cat);
                double[] arr = entry.getValue();
                StringBuilder sb = new StringBuilder();
                int maxIdx = 0;
                double maxVal = -1;
                for (int i = 0; i < 8; i++) {
                    double meanNorm = (arr[i] / count) * 100;
                    if (meanNorm > maxVal) {
                        maxVal = meanNorm;
                        maxIdx = i;
                    }
                    sb.append(String.format("%s:%.1f%% ", emoNames[i], meanNorm));
                }
                System.out.printf("[%s] (%d docs) -> Dominante: %s%n  %s%n", cat, count, emoNames[maxIdx], sb.toString());
            }
        }
        System.out.println("=======================================================\n");
    }
}