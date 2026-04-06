package com.nlp.pipeline.vectorization;

import com.nlp.pipeline.model.DocumentResult;

import java.util.*;
import java.util.stream.Collectors;

public class VocabularyBuilder {

    public static List<String> buildVocabulary(List<DocumentResult> docs, int maxSize) {

        Map<String, Integer> freq = new HashMap<>();

        for (DocumentResult doc : docs) {
            for (String token : doc.tokens) {
                freq.put(token, freq.getOrDefault(token, 0) + 1);
            }
        }

        return freq.entrySet().stream()
                .sorted((a, b) -> b.getValue() - a.getValue())
                .limit(maxSize)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
}