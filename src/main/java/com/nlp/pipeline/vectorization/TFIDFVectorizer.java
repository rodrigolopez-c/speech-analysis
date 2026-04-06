package com.nlp.pipeline.vectorization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TFIDFVectorizer {

    public static List<double[]> transform(List<double[]> bowMatrix) {

        int numDocs = bowMatrix.size();
        int vocabSize = bowMatrix.get(0).length;

        double[] df = new double[vocabSize];

        // Document Frequency
        for (double[] doc : bowMatrix) {
            for (int i = 0; i < vocabSize; i++) {
                if (doc[i] > 0) df[i]++;
            }
        }

        List<double[]> tfidfMatrix = new ArrayList<>();

        for (double[] doc : bowMatrix) {

            double[] tfidf = new double[vocabSize];
            double sum = Arrays.stream(doc).sum();

            for (int i = 0; i < vocabSize; i++) {

                double tf = doc[i] / (sum == 0 ? 1 : sum);
                double idf = Math.log((double) numDocs / (1 + df[i]));

                tfidf[i] = tf * idf;
            }

            tfidfMatrix.add(tfidf);
        }

        return tfidfMatrix;
    }
}