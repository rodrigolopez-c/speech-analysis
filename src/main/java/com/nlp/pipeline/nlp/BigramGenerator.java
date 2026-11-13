package com.nlp.pipeline.nlp;

import java.util.ArrayList;
import java.util.List;

public class BigramGenerator {

    public static List<String> generateBigrams(List<String> tokens) {

        List<String> bigrams = new ArrayList<>();

        for (int i = 0; i < tokens.size() - 1; i++) {
            String w1 = tokens.get(i);
            String w2 = tokens.get(i + 1);

            // Omitir bigramas repetidos como capitulo_capitulo o anexo_anexo
            if (w1.equalsIgnoreCase(w2)) continue;

            String bigram = w1 + "_" + w2;
            bigrams.add(bigram);
        }

        return bigrams;
    }
}