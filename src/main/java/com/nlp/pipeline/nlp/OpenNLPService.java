package com.nlp.pipeline.nlp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

public class OpenNLPService {

    private DoccatModel model;

    // Entrenamiento
    public void trainModel(List<String> texts, List<String> labels) {

        try {

            File trainingFile = new File("src/main/resources/training.txt");

            int validSamples = 0;
            try (BufferedWriter writer = java.nio.file.Files.newBufferedWriter(trainingFile.toPath(), StandardCharsets.UTF_8)) {
                for (int i = 0; i < texts.size(); i++) {
                    String text = texts.get(i) != null ? texts.get(i).trim() : "";
                    if (text.isEmpty()) {
                        continue;
                    }
                    String cleanLabel = labels.get(i) != null ? labels.get(i).trim().replaceAll("\\s+", "_") : "Unknown";
                    if (cleanLabel.isEmpty()) {
                        cleanLabel = "Unknown";
                    }
                    writer.write(cleanLabel + "\t" + text);
                    writer.newLine();
                    validSamples++;
                }
            }

            if (validSamples == 0) {
                System.err.println("No hay ejemplos válidos para entrenar el modelo OpenNLP.");
                return;
            }

            InputStreamFactory dataIn = new MarkableFileInputStreamFactory(trainingFile);
            ObjectStream<String> lineStream = new PlainTextByLineStream(dataIn, StandardCharsets.UTF_8);
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

            TrainingParameters params = new TrainingParameters();
            params.put(TrainingParameters.ITERATIONS_PARAM, "100");
            params.put(TrainingParameters.CUTOFF_PARAM, "1");

            model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());

            System.out.println("Modelo entrenado correctamente con " + validSamples + " muestras.");

        } catch (IOException e) {
            throw new RuntimeException("Error entrenando modelo", e);
        }
    }

    // Predicción
    public String predict(String text) {
        if (model == null) {
            return "Desconocido";
        }
        DocumentCategorizerME categorizer = new DocumentCategorizerME(model);

        String[] tokens = (text != null && !text.trim().isEmpty()) ? text.trim().split("\\s+") : new String[]{"empty"};
        double[] outcomes = categorizer.categorize(tokens);
        return categorizer.getBestCategory(outcomes);
    }
}