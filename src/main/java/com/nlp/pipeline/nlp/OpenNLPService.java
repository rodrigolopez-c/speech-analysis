package com.nlp.pipeline.nlp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
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

            File trainingFile = new File("src\\main\\resources\\training.txt");

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(trainingFile))) {
                for (int i = 0; i < texts.size(); i++) {
                    writer.write(labels.get(i) + "\t" + texts.get(i));
                    writer.newLine();
                }
            }

            InputStreamFactory dataIn = new MarkableFileInputStreamFactory(trainingFile);
            ObjectStream<String> lineStream = new PlainTextByLineStream(dataIn, StandardCharsets.UTF_8);
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

            TrainingParameters params = new TrainingParameters();
            params.put(TrainingParameters.ITERATIONS_PARAM, "100");
            params.put(TrainingParameters.CUTOFF_PARAM, "1");

            model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());

            System.out.println("Modelo entrenado correctamente");

        } catch (IOException e) {
            throw new RuntimeException("Error entrenando modelo", e);
        }
    }

    // Predicción
    public String predict(String text) {
        DocumentCategorizerME categorizer = new DocumentCategorizerME(model);

        String[] tokens = text.split("\\s+");
        double[] outcomes = categorizer.categorize(tokens);
        return categorizer.getBestCategory(outcomes);
    }
}