package com.nlp.pipeline.pipeline;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.nlp.BigramGenerator;
import com.nlp.pipeline.nlp.CoreNLPService;
import com.nlp.pipeline.nlp.NRCEmotionService;
import com.nlp.pipeline.nlp.TextNormalizer;
import com.nlp.pipeline.utils.StopWordsLoader;

public class DocumentProcessor {

    private final CoreNLPService coreNLPService;
    private final NRCEmotionService nrcEmotionService;
    private final Set<String> stopWords;

    public DocumentProcessor() {
        this.coreNLPService = new CoreNLPService();
        this.nrcEmotionService = new NRCEmotionService();
        this.stopWords = StopWordsLoader.loadStopWords("stopwords.txt");
    }

    public void processDocuments(List<DocumentResult> documents) {
        int total = documents.size();
        AtomicInteger count = new AtomicInteger(0);

        IntStream.range(0, total).parallel().forEach(i -> {
            DocumentResult doc = documents.get(i);
            int current = count.incrementAndGet();
            if (current % 25 == 0 || current == 1 || current == total) {
                System.out.println("  [" + current + "/" + total + "] Procesando: " + doc.category + " (" + doc.fileName + ")...");
            }

            // CoreNLP
            Map<String, Object> result = coreNLPService.processText(doc.content);

            synchronized (doc) {
                doc.tokens = (List<String>) result.get("tokens");
                doc.posTags = (List<String>) result.get("pos");
                doc.namedEntities = (List<String>) result.get("ner");
                doc.sentenceCount = (int) result.get("sentenceCount");

                // Limpieza (tokens + POS)
                TextNormalizer.cleanDocument(
                        doc.tokens,
                        doc.posTags,
                        stopWords
                );

                // Análisis de Emociones NRC integrado con POS tags de StanfordNLP
                doc.emotionProfile = nrcEmotionService.analyzeTokensWithPOS(doc.tokens, doc.posTags);
                doc.sentimentScore = nrcEmotionService.toPolarityScore(doc.emotionProfile);

                // Generar bigramas
                List<String> bigrams = BigramGenerator.generateBigrams(doc.tokens);

                // Agregar bigramas a tokens
                doc.tokens.addAll(bigrams);
            }
        });
    }
}