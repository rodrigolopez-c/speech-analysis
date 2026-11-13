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
        props.setProperty("ner.applyNumericClassifiers", "false");
        props.setProperty("ner.useSUTime", "false");
        props.setProperty("ner.applyFineGrained", "false");

        this.pipeline = new StanfordCoreNLP(props);
    }

    public Map<String, Object> processText(String text) {
        // Para acelerar el procesamiento de documentos gigantes de 30k+ caracteres,
        // limitamos la anotación profunda a los primeros 8,000 caracteres por documento
        String textToProcess = text;
        if (textToProcess.length() > 8000) {
            textToProcess = textToProcess.substring(0, 8000);
        }

        Annotation document = new Annotation(textToProcess);
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