package com.nlp.pipeline.nlp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.TrainingParameters;

public class SentimentService {

    private DoccatModel model;
    private final Set<String> positiveLexicon = new HashSet<>(Arrays.asList(
            "éxito", "exito", "logro", "logros", "desarrollo", "crecimiento", "avance", "avances",
            "positivo", "mejor", "excelente", "compromiso", "futuro", "calidad", "innovación",
            "innovacion", "bien", "beneficio", "fortaleza", "superación", "superacion", "gratitud",
            "apoyo", "confianza", "esperanza", "triunfo", "oportunidad", "oportunidades",
            "success", "growth", "positive", "best", "excellent", "future", "quality", "support"
    ));
    private final Set<String> negativeLexicon = new HashSet<>(Arrays.asList(
            "problema", "problemas", "crisis", "difícil", "dificil", "riesgo", "riesgos",
            "pérdida", "perdida", "negativo", "conflicto", "daño", "dano", "falla", "fallas",
            "error", "errores", "déficit", "deficit", "violencia", "amenaza", "fracaso",
            "preocupación", "preocupacion", "caída", "caida", "desafío", "desafio",
            "problem", "crisis", "difficult", "risk", "loss", "negative", "error", "failure"
    ));

    public SentimentService() {
        trainModel();
    }

    private void trainModel() {
        String csvPath = "src/main/resources/Datasets/SentimentAnalysisDataset/train.csv";
        File csvFile = new File(csvPath);

        if (!csvFile.exists()) {
            System.out.println("Dataset de sentimientos no encontrado en: " + csvPath + ". Se utilizará el analizador léxico por defecto.");
            return;
        }

        System.out.println("Entrenando modelo de sentimientos desde CSV...");
        List<DocumentSample> samples = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue; // Skip header
                }

                int lastCommaIndex = line.lastIndexOf(',');
                if (lastCommaIndex == -1) continue;

                String text = line.substring(0, lastCommaIndex).trim();
                String category = line.substring(lastCommaIndex + 1).trim();

                if (text.startsWith("\"") && text.endsWith("\"") && text.length() > 1) {
                    text = text.substring(1, text.length() - 1);
                }

                String[] tokens = text.split("\\s+");
                samples.add(new DocumentSample(category, tokens));
            }

            if (!samples.isEmpty()) {
                ObjectStream<DocumentSample> sampleStream = ObjectStreamUtils.createObjectStream(samples);

                TrainingParameters params = new TrainingParameters();
                params.put(TrainingParameters.ITERATIONS_PARAM, "20");
                params.put(TrainingParameters.CUTOFF_PARAM, "2");

                model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());
                System.out.println("Modelo de sentimientos entrenado correctamente.");
            }

        } catch (IOException e) {
            System.err.println("Advertencia entrenando modelo de sentimientos: " + e.getMessage() + ". Usando léxico de respaldo.");
        }
    }

    public double analyzeSentiment(String text) {
        if (text == null || text.trim().isEmpty()) {
            return 2.0;
        }

        if (model != null) {
            DocumentCategorizerME categorizer = new DocumentCategorizerME(model);
            String[] tokens = text.split("\\s+");
            double[] outcomes = categorizer.categorize(tokens);
            String category = categorizer.getBestCategory(outcomes);

            switch (category.toLowerCase()) {
                case "positive":
                    return 3.0;
                case "neutral":
                    return 2.0;
                case "negative":
                    return 1.0;
                default:
                    return 2.0;
            }
        }

        // Analizador léxico por defecto
        String[] words = text.toLowerCase().split("[\\s\\p{Punct}]+");
        int posCount = 0;
        int negCount = 0;

        for (String word : words) {
            if (positiveLexicon.contains(word)) {
                posCount++;
            } else if (negativeLexicon.contains(word)) {
                negCount++;
            }
        }

        if (posCount > negCount) {
            return 3.0; // Positivo
        } else if (negCount > posCount) {
            return 1.0; // Negativo
        } else {
            return 2.0; // Neutral
        }
    }
}