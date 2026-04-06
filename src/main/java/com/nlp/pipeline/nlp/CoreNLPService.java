package com.nlp.pipeline.nlp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class CoreNLPService {

    private final StanfordCoreNLP pipeline;

    public CoreNLPService() {
        Properties props = new Properties();

        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
        props.setProperty("tokenize.language", "en");

        this.pipeline = new StanfordCoreNLP(props);
    }

    public Map<String, Object> processText(String text) {

        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        List<String> tokens = new ArrayList<>();
        List<String> posTags = new ArrayList<>();
        List<String> nerTags = new ArrayList<>();

        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

        for (CoreMap sentence : sentences) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

                tokens.add(token.word());
                posTags.add(token.get(CoreAnnotations.PartOfSpeechAnnotation.class));
                nerTags.add(token.get(CoreAnnotations.NamedEntityTagAnnotation.class));
            }
        }

        Map<String, Object> result = new HashMap<>();
        result.put("tokens", tokens);
        result.put("pos", posTags);
        result.put("ner", nerTags);
        result.put("sentenceCount", sentences.size());

        return result;
    }
}