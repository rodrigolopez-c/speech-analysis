package com.nlp.pipeline.model;

import java.util.List;

public class DocumentResult {

    public String category;
    public String fileName;
    public String content;
    public int sentenceCount;
    public double sentimentScore;

    public List<String> tokens;
    public List<String> posTags;
    public List<String> namedEntities;

    public EmotionProfile emotionProfile;
    public double sarcasmScore = -1.0;
    public double metaphorRatio = -1.0;

    public DocumentResult(String category, String fileName, String content) {
        this.category = category;
        this.fileName = fileName;
        this.content = content;
    }
}