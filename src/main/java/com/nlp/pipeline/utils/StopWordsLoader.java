package com.nlp.pipeline.utils;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;

public class StopWordsLoader {

    public static Set<String> loadStopWords(String fileName) {
        Set<String> stopWords = new HashSet<>();

        try {
            InputStream is = StopWordsLoader.class
                    .getClassLoader()
                    .getResourceAsStream(fileName);

            BufferedReader reader = new BufferedReader(new InputStreamReader(is));

            String line;
            while ((line = reader.readLine()) != null) {
                stopWords.add(line.trim().toLowerCase());
            }

        } catch (Exception e) {
            throw new RuntimeException("Error cargando stopwords", e);
        }

        return stopWords;
    }
}