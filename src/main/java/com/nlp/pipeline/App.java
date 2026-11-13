package com.nlp.pipeline;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.nlp.OpenNLPService;
import com.nlp.pipeline.nlp.PythonPipelineBridge;
import com.nlp.pipeline.nlp.SentimentService;
import com.nlp.pipeline.pipeline.ClassificationSummary;
import com.nlp.pipeline.pipeline.CorpusManager;
import com.nlp.pipeline.pipeline.CorpusSummary;
import com.nlp.pipeline.pipeline.DocumentExporter;
import com.nlp.pipeline.pipeline.DocumentProcessor;
import com.nlp.pipeline.pipeline.SentimentSummary;
import com.nlp.pipeline.vectorization.BagOfWords;
import com.nlp.pipeline.vectorization.PCAReducer;
import com.nlp.pipeline.vectorization.TFIDFVectorizer;
import com.nlp.pipeline.vectorization.VocabularyBuilder;

public class App {

    public static void main(String[] args) {

        System.out.println("=========================================================");
        System.out.println("       PIPELINE NLP - EVALUACIÓN COMPARATIVA DE DATASETS   ");
        System.out.println("=========================================================\n");

        String[] datasetPaths;

        if (args.length > 0) {
            datasetPaths = args;
        } else {
            datasetPaths = new String[]{
                "src/main/resources/datasets/diario_debates_senado_trafilatura.csv"
            };
        }

        int limitPerCategory = -1; // Cargar todos los documentos

        for (String datasetPath : datasetPaths) {
            File file = new File(datasetPath);
            if (!file.exists()) {
                System.err.println("Archivo no encontrado: " + datasetPath);
                continue;
            }

            String datasetName = file.getName();
            System.out.println("\n#########################################################");
            System.out.println(" PROCESANDO DATASET: " + datasetName);
            System.out.println("#########################################################\n");

            long startTime = System.currentTimeMillis();

            System.out.println("Cargando dataset desde: " + datasetPath + "...");
            CorpusManager corpusManager = new CorpusManager(datasetPath, limitPerCategory);
            List<DocumentResult> documents = corpusManager.loadCorpus();

            if (documents.isEmpty()) {
                System.err.println("No se encontraron documentos para procesar en " + datasetPath);
                continue;
            }

            System.out.println("Documentos cargados exitosamente: " + documents.size());

            // POS tagging y preprocesamiento
            System.out.println("\nProcesando documentos (Tokenización, POS Tagging, NER, Normalización)...");
            long procStart = System.currentTimeMillis();
            DocumentProcessor processor = new DocumentProcessor();
            processor.processDocuments(documents);
            long procEnd = System.currentTimeMillis();

            System.out.println("\n--- Resumen Estadístico del Corpus ---");
            CorpusSummary.printSummary(documents);

            // Vocabulario (máx 5000 palabras)
            List<String> vocabulary = VocabularyBuilder.buildVocabulary(documents, 5000);
            System.out.println("\nTamaño del vocabulario generado: " + vocabulary.size());

            // Vectorización Bag of Words
            List<double[]> bowMatrix = BagOfWords.transform(documents, vocabulary);

            // Vectorización TF-IDF
            List<double[]> tfidfMatrix = TFIDFVectorizer.transform(bowMatrix);

            System.out.println("\n=== Vectorización ===");
            System.out.println("Matriz TF-IDF: " + documents.size() + " x " + (tfidfMatrix.isEmpty() ? 0 : tfidfMatrix.get(0).length));

            // Reducción de dimensionalidad con PCA
            int numComponents = 2; // para visualización
            if (!tfidfMatrix.isEmpty() && tfidfMatrix.get(0).length > 0) {
                double[][] pcaResult = PCAReducer.applyPCA(tfidfMatrix, numComponents);
                System.out.println("\nPCA completado:");
                System.out.println("Dimensión original: " + tfidfMatrix.get(0).length);
                System.out.println("Dimensión reducida: " + pcaResult[0].length);
            }

            // Clasificación de Autor
            System.out.println("\n=== Clasificación (Detección de Autor) ===");
            OpenNLPService classifier = new OpenNLPService();

            List<String> texts = new ArrayList<>();
            List<String> labels = new ArrayList<>();

            for (DocumentResult doc : documents) {
                String text = String.join(" ", doc.tokens);
                texts.add(text);
                labels.add(doc.category);
            }

            classifier.trainModel(texts, labels);
            ClassificationSummary.printSummary(documents, classifier);

            // Análisis de Sentimientos y Emociones (NRC EmoLex + StanfordNLP)
            System.out.println("\n=== Análisis de Sentimientos y Emociones (NRC EmoLex) ===");
            SentimentSummary.printSummary(documents);

            // Exportar datos procesados con vectores de emoción para los módulos de Sarcasmo y Metáforas
            String exportPath = "target/processed_" + datasetName.replace(".xlsx", ".csv");
            if (!exportPath.endsWith(".csv")) exportPath += ".csv";
            DocumentExporter.exportToCSV(documents, exportPath);

            // Módulos Avanzados (Sarcasmo y Metáforas vía Python Bridge)
            System.out.println("\n=== Módulos Avanzados (Sarcasmo y Metáforas vía Python Bridge) ===");
            PythonPipelineBridge bridge = new PythonPipelineBridge();
            bridge.executePythonModule("src/python/pipeline_runner.py", "--csv", exportPath, "--sample", "-1");

            long endTime = System.currentTimeMillis();
            System.out.println("\nTiempo total de ejecución para " + datasetName + ": " + (endTime - startTime) + " ms (Preprocesamiento: " + (procEnd - procStart) + " ms)");
            System.out.println("=========================================================\n");
        }

        System.out.println("Pipeline ejecutado correctamente para todos los datasets.");
    }
}