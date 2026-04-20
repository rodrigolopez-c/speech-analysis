package com.nlp.pipeline.nlp;

import java.util.ArrayList;
import java.util.List;

import com.nlp.pipeline.utils.SentimentDatasetLoader;

public class SentimentTrainer {

    public static void train(OpenNLPService classifier) {

        String path = "src/main/resources/Datasets/SentimentAnalysisDataset/train.csv";

        List<String[]> data = SentimentDatasetLoader.load(path, 10000);

        List<String> texts = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        for (String[] row : data) {

            String label = row[0];
            String text = row[1];

            // convertir etiqueta
            if (label.equals("0")) label = "negativo";
            else if (label.equals("4")) label = "positivo";
            else continue;

            texts.add(text);
            labels.add(label);
        }

        classifier.trainModel(texts, labels);
    }
}