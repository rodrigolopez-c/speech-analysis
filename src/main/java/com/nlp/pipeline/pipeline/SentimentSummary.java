package com.nlp.pipeline.pipeline;

import java.util.List;

import com.nlp.pipeline.model.DocumentResult;

public class SentimentSummary {

    public static void printSummary(List<DocumentResult> docs) {

        double total = 0;
        int positive = 0;
        int neutral = 0;
        int negative = 0;

        for (DocumentResult doc : docs) {

            if (doc.sentimentScore == 0) continue;

            double score = doc.sentimentScore;
            total += score;

            if (score >= 3) positive++;
            else if (score >= 2) neutral++;
            else negative++;
        }

        double avg = total / docs.size();

        System.out.println("\n===== SENTIMIENTO =====");

        System.out.printf("Sentimiento promedio: %.2f%n", avg);

        System.out.println("\nDistribución:");
        System.out.println("Positivos: " + positive);
        System.out.println("Neutrales: " + neutral);
        System.out.println("Negativos: " + negative);

    }
}