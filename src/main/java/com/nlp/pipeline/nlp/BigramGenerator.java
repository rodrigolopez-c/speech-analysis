package com.nlp.pipeline.nlp;

import java.util.ArrayList;
import java.util.List;

public class BigramGenerator {

    public static List<String> generateBigrams(List<String> tokens) {

        List<String> bigrams = new ArrayList<>();

        for (int i = 0; i < tokens.size() - 1; i++) {
            String bigram = tokens.get(i) + "_" + tokens.get(i + 1);
            bigrams.add(bigram);
        }

        return bigrams;
    }
}