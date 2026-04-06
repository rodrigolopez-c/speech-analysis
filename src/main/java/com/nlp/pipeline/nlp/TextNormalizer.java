package com.nlp.pipeline.nlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class TextNormalizer {

    public static void cleanDocument(
            List<String> tokens,
            List<String> posTags,
            Set<String> stopWords
    ) {

        List<String> cleanedTokens = new ArrayList<>();
        List<String> cleanedPOS = new ArrayList<>();

        for (int i = 0; i < tokens.size(); i++) {

            String token = tokens.get(i);
            String pos = posTags.get(i);

            // Normalización
            token = token.toLowerCase().trim();

            // eliminar tokens vacíos o muy cortos
            if (token.length() < 2) continue;

            // eliminar números
            if (token.matches("\\d+")) continue;

            // eliminar puntuación
            if (!token.matches("[a-záéíóúñü]+")) continue;

            // eliminar stopwords
            if (stopWords.contains(token)) continue;

            cleanedTokens.add(token);
            cleanedPOS.add(pos);
        }

        tokens.clear();
        tokens.addAll(cleanedTokens);

        posTags.clear();
        posTags.addAll(cleanedPOS);
    }
}