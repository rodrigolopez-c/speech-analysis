package com.nlp.pipeline.nlp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.TrainingParameters;

public class SentimentService {

    private DoccatModel model;

    public SentimentService() {
        trainModel();
    }

    private void trainModel() {
        System.out.println("Entrenando modelo de sentimientos...");
        String csvPath = "src/main/resources/Datasets/SentimentAnalysisDataset/train.csv";
        List<DocumentSample> samples = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue; // Skip header
                }
                
                int lastCommaIndex = line.lastIndexOf(',');
                if (lastCommaIndex == -1) continue;

                String text = line.substring(0, lastCommaIndex).trim();
                String category = line.substring(lastCommaIndex + 1).trim();

                // Clean quotes if present
                if (text.startsWith("\"") && text.endsWith("\"") && text.length() > 1) {
                    text = text.substring(1, text.length() - 1);
                }

                String[] tokens = text.split("\\s+");
                samples.add(new DocumentSample(category, tokens));
            }

            ObjectStream<DocumentSample> sampleStream = ObjectStreamUtils.createObjectStream(samples);

            TrainingParameters params = new TrainingParameters();
            params.put(TrainingParameters.ITERATIONS_PARAM, "20"); // Using 20 for faster startup
            params.put(TrainingParameters.CUTOFF_PARAM, "2");

            model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());

            System.out.println("Modelo de sentimientos entrenado correctamente.");

        } catch (IOException e) {
            throw new RuntimeException("Error entrenando modelo de sentimientos", e);
        }
    }

    public double analyzeSentiment(String text) {
        if (model == null) {
            return 0.0;
        }

        DocumentCategorizerME categorizer = new DocumentCategorizerME(model);
        String[] tokens = text.split("\\s+");
        double[] outcomes = categorizer.categorize(tokens);
        String category = categorizer.getBestCategory(outcomes);

        switch (category.toLowerCase()) {
            case "positive":
                return 3.0;
            case "neutral":
                return 2.0;
            case "negative":
                return 1.0;
            default:
                return 2.0;
        }
    }
}