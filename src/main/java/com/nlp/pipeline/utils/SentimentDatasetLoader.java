package com.nlp.pipeline.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SentimentDatasetLoader {

    public static List<String[]> load(String filePath, int limit) {

        List<String[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {

            String line;
            int count = 0;

            while ((line = br.readLine()) != null) {

                if (limit > 0 && count >= limit) break;

                String[] parts = line.split(",", 6);

                if (parts.length < 6) continue;

                String label = parts[0];
                String text = parts[5];

                data.add(new String[]{label, text});

                count++;
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return data;
    }
}