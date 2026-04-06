package com.nlp.pipeline.nlp;

import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class SentimentService {

    private final StanfordCoreNLP pipeline;

    public SentimentService() {
        Properties props = new Properties();

        // pipeline necesario para sentimiento
        props.setProperty("annotators", "tokenize,ssplit,parse,sentiment");
        props.setProperty("tokenize.language", "en");

        this.pipeline = new StanfordCoreNLP(props);
    }

    public double analyzeSentiment(String text) {

        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        List<CoreMap> sentences =
                document.get(CoreAnnotations.SentencesAnnotation.class);

        double total = 0;
        int count = 0;

        int maxSentences = 1;

        for (int i = 0; i < sentences.size() && i < maxSentences; i++) {

            CoreMap sentence = sentences.get(i);

            Tree tree = sentence.get(
                    SentimentCoreAnnotations.SentimentAnnotatedTree.class
            );

            int score = RNNCoreAnnotations.getPredictedClass(tree);

            total += score;
            count++;
        }

        return count == 0 ? 0 : total / count;
    }
}