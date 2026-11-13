package com.nlp.pipeline.pipeline;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.List;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.model.EmotionProfile;

/**
 * Exportador de resultados del pipeline en Java para interoperabilidad con los módulos
 * avanzados en Python (Detección de Sarcasmo con Transformers/Scikit-Learn y Detección de Metáforas con BiLSTM/WordNet).
 */
public class DocumentExporter {

    public static void exportToCSV(List<DocumentResult> documents, String outputPath) {
        File file = new File(outputPath);
        file.getParentFile().mkdirs();

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8))) {
            // Cabecera CSV
            writer.write("id,category,fileName,sentenceCount,predominantEmotion,avgIntensity," +
                    "anger,disgust,fear,sadness,surprise,anticipation,joy,trust,tokens,posTags,sampleText\n");

            for (int i = 0; i < documents.size(); i++) {
                DocumentResult doc = documents.get(i);
                EmotionProfile p = doc.emotionProfile != null ? doc.emotionProfile : new EmotionProfile();

                String safeTokens = doc.tokens != null ? String.join(" ", doc.tokens) : "";
                if (safeTokens.length() > 2000) {
                    safeTokens = safeTokens.substring(0, 2000);
                }
                safeTokens = safeTokens.replace("\"", "\"\"");

                String safePos = doc.posTags != null ? String.join(" ", doc.posTags) : "";
                if (safePos.length() > 2000) {
                    safePos = safePos.substring(0, 2000);
                }
                safePos = safePos.replace("\"", "\"\"");

                String sampleText = doc.content != null ? doc.content.replace("\n", " ").replace("\r", " ") : "";
                if (sampleText.length() > 8500) {
                    // En sesiones parlamentarias extensas, los primeros ~3500 caracteres son pase de lista y quórum;
                    // combinamos el encabezado con un bloque central de debate en tribuna
                    sampleText = sampleText.substring(0, 1500) + " " + sampleText.substring(4000, 8500);
                } else if (sampleText.length() > 6000) {
                    sampleText = sampleText.substring(0, 6000);
                }
                sampleText = sampleText.replace("\"", "\"\"");

                writer.write(String.format("\"%d\",\"%s\",\"%s\",%d,\"%s\",%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,\"%s\",\"%s\",\"%s\"\n",
                        i,
                        doc.category != null ? doc.category.replace("\"", "\"\"") : "",
                        doc.fileName != null ? doc.fileName.replace("\"", "\"\"") : "",
                        doc.sentenceCount,
                        p.predominantEmotion,
                        p.averageIntensity,
                        p.anger, p.disgust, p.fear, p.sadness, p.surprise, p.anticipation, p.joy, p.trust,
                        safeTokens,
                        safePos,
                        sampleText
                ));
            }

            System.out.println("Documentos y características exportados exitosamente a: " + file.getAbsolutePath());
        } catch (Exception e) {
            System.err.println("Error al exportar documentos: " + e.getMessage());
        }
    }
}
