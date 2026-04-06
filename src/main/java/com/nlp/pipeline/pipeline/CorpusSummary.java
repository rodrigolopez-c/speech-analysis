package com.nlp.pipeline.pipeline;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nlp.pipeline.model.DocumentResult;

public class CorpusSummary {

    public static void printSummary(List<DocumentResult> docs) {

        Map<String, Integer> docsPerCategory = new HashMap<>();
        Map<String, Integer> tokenFreq = new HashMap<>();
        Map<String, Integer> posFreq = new HashMap<>();
        Map<String, Integer> nerFreq = new HashMap<>();
        Map<String, Integer> bigramFreq = new HashMap<>();

        int totalDocs = docs.size();
        int totalChars = 0;
        int totalTokens = 0;
        int totalSentences = 0;

        for (DocumentResult doc : docs) {

            docsPerCategory.put(
                    doc.category,
                    docsPerCategory.getOrDefault(doc.category, 0) + 1
            );

            totalChars += doc.content.length();
            totalTokens += doc.tokens.size();
            totalSentences += doc.sentenceCount;

            // tokens
            for (String token : doc.tokens) {
                tokenFreq.put(token, tokenFreq.getOrDefault(token, 0) + 1);
            }

            // POS
            for (String pos : doc.posTags) {
                posFreq.put(pos, posFreq.getOrDefault(pos, 0) + 1);
            }

            // NER
            for (String ner : doc.namedEntities) {
                if (!ner.equals("O")) { // O = no entidad
                    nerFreq.put(ner, nerFreq.getOrDefault(ner, 0) + 1);
                }
            }

            // bigramas
            for (String token : doc.tokens) {
                if (token.contains("_")) {
                    bigramFreq.put(token, bigramFreq.getOrDefault(token, 0) + 1);
                }
            }
        }

        System.out.println("\n========== CORPUS SUMMARY ==========");
        System.out.println("Total documentos: " + totalDocs);
        System.out.println("Total caracteres: " + totalChars);
        System.out.println("Total tokens: " + totalTokens);
        System.out.println("Total oraciones: " + totalSentences);

        /* 
        System.out.println("\nDocumentos por categoría:");
        docsPerCategory.forEach((cat, count) ->
                System.out.println("- " + cat + ": " + count)
        );
        */

        // Top tokens
        System.out.println("\n=== Top 20 tokens más frecuentes ===");
        tokenFreq.entrySet().stream()
                .sorted((a, b) -> b.getValue() - a.getValue())
                .limit(20)
                .forEach(entry ->
                        System.out.println(entry.getKey() + ": " + entry.getValue())
                );

        // Top POS
        System.out.println("\n=== Top 20 POS más frecuentes (%) ===");

        int totalPos = posFreq.values().stream().mapToInt(i -> i).sum();

        posFreq.entrySet().stream()
                .sorted((a, b) -> b.getValue() - a.getValue())
                .limit(20)
                .forEach(entry -> {
                    double percentage = (entry.getValue() * 100.0) / totalPos;
                    System.out.printf("%s: %.2f%%%n",
                            entry.getKey(),
                            percentage);
                });

        // Top NER
        System.out.println("\n=== Top 20 NER más frecuentes ===");

        nerFreq.entrySet().stream()
                .sorted((a, b) -> b.getValue() - a.getValue())
                .limit(20)
                .forEach(entry ->
                        System.out.println(entry.getKey() + ": " + entry.getValue())
                );

        // Top bigrams
        System.out.println("\n=== Top 20 bigramas más frecuentes ===");

        bigramFreq.entrySet().stream()
                .sorted((a, b) -> b.getValue() - a.getValue())
                .limit(20)
                .forEach(entry ->
                        System.out.println(entry.getKey() + ": " + entry.getValue())
                );
    }

}