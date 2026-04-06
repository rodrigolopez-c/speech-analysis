package com.nlp.pipeline.pipeline;

import java.util.List;
import java.util.Map;
import java.util.Set;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.nlp.BigramGenerator;
import com.nlp.pipeline.nlp.CoreNLPService;
import com.nlp.pipeline.nlp.TextNormalizer;
import com.nlp.pipeline.utils.StopWordsLoader;

public class DocumentProcessor {

    private final CoreNLPService coreNLPService;
    private final Set<String> stopWords;

    public DocumentProcessor() {
        this.coreNLPService = new CoreNLPService();
        this.stopWords = StopWordsLoader.loadStopWords("stopwords.txt");
    }

    public void processDocuments(List<DocumentResult> documents) {

        for (DocumentResult doc : documents) {

            // CoreNLP
            Map<String, Object> result =
                    coreNLPService.processText(doc.content);

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

            // Generar bigramas
            List<String> bigrams = BigramGenerator.generateBigrams(doc.tokens);

            // Agregar bigramas a tokens
            doc.tokens.addAll(bigrams);
        }
    }
}