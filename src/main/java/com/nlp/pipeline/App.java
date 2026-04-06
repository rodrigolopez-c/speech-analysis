package com.nlp.pipeline;

import java.util.ArrayList;
import java.util.List;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.nlp.OpenNLPService;
import com.nlp.pipeline.nlp.SentimentService;
import com.nlp.pipeline.pipeline.ClassificationSummary;
import com.nlp.pipeline.pipeline.CorpusManager;
import com.nlp.pipeline.pipeline.CorpusSummary;
import com.nlp.pipeline.pipeline.DocumentProcessor;
import com.nlp.pipeline.pipeline.SentimentSummary;
import com.nlp.pipeline.vectorization.BagOfWords;
import com.nlp.pipeline.vectorization.PCAReducer;
import com.nlp.pipeline.vectorization.TFIDFVectorizer;
import com.nlp.pipeline.vectorization.VocabularyBuilder;

public class App {

    public static void main(String[] args) {

        String path = "src/main/resources/documents/textos_secciones";

        // Para pruebas, limitar el número de documentos por categoría ('-1' para cargar todo)
        int limitPerCategory = -1;

        CorpusManager corpusManager = new CorpusManager(path, limitPerCategory);
        List<DocumentResult> documents = corpusManager.loadCorpus();

        // POS tagging
        DocumentProcessor processor = new DocumentProcessor();
        processor.processDocuments(documents);

        CorpusSummary.printSummary(documents);

        // Vocabulario (máx 5000 palabras)
        List<String> vocabulary = VocabularyBuilder.buildVocabulary(documents, 5000);

        // Vectorización Bag of Words
        List<double[]> bowMatrix = BagOfWords.transform(documents, vocabulary);

        // Vectorización TF-IDF
        List<double[]> tfidfMatrix = TFIDFVectorizer.transform(bowMatrix);

        System.out.println("\n=== Vectorización ===");

        System.out.println("Matriz TF-IDF: " + documents.size() + " x " + tfidfMatrix.get(0).length);

        // Reducción de dimensionalidad con PCA

        int numComponents = 2; // para visualización

        double[][] pcaResult = PCAReducer.applyPCA(tfidfMatrix, numComponents);

        System.out.println("\nPCA completado:");
        System.out.println("Dimensión original: " + tfidfMatrix.get(0).length);
        System.out.println("Dimensión reducida: " + pcaResult[0].length);

        // Clasificación

        OpenNLPService classifier = new OpenNLPService();

        // preparar datos para entrenamiento
        List<String> texts = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        for (DocumentResult doc : documents) {
            String text = String.join(" ", doc.tokens);
            texts.add(text);
            labels.add(doc.category);
        }

        // entrenar modelo
        classifier.trainModel(texts, labels);

        ClassificationSummary.printSummary(documents, classifier);

        // Sentiment analysis

        SentimentService sentimentService = new SentimentService();

        int counter = 0;

        for (DocumentResult doc : documents) {

            String text = String.join(" ", doc.tokens);

            if (text.length() > 300) {
                text = text.substring(0, 300);
            }

            doc.sentimentScore = sentimentService.analyzeSentiment(text);

            counter++;

            if (counter % 10 == 0) {
                System.gc();
            }
        }

        SentimentSummary.printSummary(documents);

        // Confirmación
        System.out.println("\nPipeline ejecutado correctamente...");
    }
}