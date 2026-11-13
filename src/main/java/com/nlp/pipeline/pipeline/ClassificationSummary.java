package com.nlp.pipeline.pipeline;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.nlp.OpenNLPService;

public class ClassificationSummary {

    public static void printSummary(List<DocumentResult> documents, OpenNLPService classifier) {

        Map<String, Integer> predictionCount = new HashMap<>();
        Map<String, Integer> correctPerClass = new HashMap<>();
        Map<String, Integer> totalPerClass = new HashMap<>();

        for (DocumentResult doc : documents) {

            String text = String.join(" ", doc.tokens);
            String predicted = classifier.predict(text).replaceAll("_", " ");
            String actual = doc.category;

            // conteo de predicciones
            predictionCount.put(predicted, predictionCount.getOrDefault(predicted, 0) + 1);

            // total por clase real
            totalPerClass.put(actual, totalPerClass.getOrDefault(actual, 0) + 1);

            // aciertos por clase
            if (predicted.equals(actual)) {
                correctPerClass.put(actual, correctPerClass.getOrDefault(actual, 0) + 1);
            }
        }

        // Accuracy por categoría
        System.out.println("\n===== CLASIFICACIÓN =====");

        System.out.println("\nAccuracy por categoría:");

        for (String category : totalPerClass.keySet()) {

            int correct = correctPerClass.getOrDefault(category, 0);
            int total = totalPerClass.get(category);

            double accuracy = (correct * 100.0) / total;

            System.out.printf("- %s: %.2f%% (%d/%d)%n",
                    category, accuracy, correct, total);
        }

        // 📊 Distribución de predicciones
        System.out.println("\nDistribución de predicciones:");

        predictionCount.forEach((category, count) ->
                System.out.println("- " + category + ": " + count)
        );

    }
}