package com.nlp.pipeline.vectorization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nlp.pipeline.model.DocumentResult;

public class BagOfWords {

    public static List<double[]> transform(List<DocumentResult> docs, List<String> vocab) {

        Map<String, Integer> vocabIndex = new HashMap<>();

        for (int i = 0; i < vocab.size(); i++) {
            vocabIndex.put(vocab.get(i), i);
        }

        List<double[]> matrix = new ArrayList<>();

        for (DocumentResult doc : docs) {

            double[] vector = new double[vocab.size()];

            for (String token : doc.tokens) {
                Integer index = vocabIndex.get(token);
                if (index != null) {
                    vector[index]++;
                }
            }

            matrix.add(vector);
        }

        return matrix;
    }
}