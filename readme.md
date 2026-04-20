# NLP Pipeline Comparison: CoreNLP vs OpenNLP

## Overview

This project implements a complete Natural Language Processing (NLP) pipeline in Java to analyze multiple `.txt` documents and compare the performance and effectiveness of two libraries:

- Stanford CoreNLP
- Apache OpenNLP

The objective is to evaluate how each library impacts different stages of an NLP pipeline and determine the most suitable option depending on the use case.

---

## Objectives

- Compare CoreNLP and OpenNLP in key NLP tasks
- Measure performance (execution time)
- Evaluate output quality (tokens, POS tags, vectorization)
- Analyze how preprocessing affects downstream tasks such as TF-IDF and PCA

---

## NLP Pipeline

The implemented pipeline includes:

Text  
→ Tokenization  
→ Sentence Segmentation  
→ POS Tagging  
→ Lemmatization (CoreNLP only)  
→ Text Cleaning (stopwords, normalization)  
→ Clasification (OpenNLP only)
→ Bigram Generation  
→ Bag of Words  
→ TF-IDF Vectorization  
→ PCA (Topic Approximation)  

---

## Comparison Scope

| Stage | CoreNLP | OpenNLP |
|------|--------|--------|
| Tokenization | Advanced | Basic |
| POS Tagging | More accurate | Model-dependent |
| Lemmatization | Yes | No |
| Performance | Slower | Faster |
| Downstream impact (TF-IDF, PCA) | Higher quality | More noise |

---

## Features Implemented

- Multi-document processing
- POS tagging comparison
- Stopword removal
- Text normalization
- Lemmatization (CoreNLP)
- Bigram generation
- Bag of Words (BoW)
- TF-IDF vectorization
- PCA for dimensionality reduction
- Corpus statistics and summaries

---

## Project Structure

src/main/java/com/nlp/pipeline/

├── model/
│ └── DocumentResult.java
│
├── nlp/
│ ├── CoreNLPService.java
│ ├── OpenNLPService.java
│ ├── BigramGenerator.java
│ ├── TextNormalizer.java
│
├── pipeline/
│ ├── CorpusManager.java
│ ├── CorpusSummary.java
│ ├── DocumentProcessor.java
│
├── vectorization/
│ ├── BagOfWords.java
│ ├── TFIDFVectorizer.java
│ ├── PCAReducer.java
│ ├── VocabularyBuilder.java
│
├── utils/
│ ├── FileUtils.java
│ ├── StopWordsLoader.java
│
└── App.java

---

## Resources

src/main/resources/

├── documents/ # Input .txt files (corpus)
├── models/ # OpenNLP models (.bin)
├── stopwords.txt # Stopword list


---

## Dependencies

- Java 23
- Maven 3.9+
- Stanford CoreNLP
- Apache OpenNLP
- Apache Commons Math (PCA)

---

## Build

```bash
mvn clean install
```

## Run

```bash
mvn exec:java "-Dexec.mainClass=com.nlp.pipeline.App" "-Dexec.jvmArgs=-Xms2g -Xmx6g"
```

## Key Findings

- Stanford CoreNLP produces higher-quality linguistic features
- Apache OpenNLP is significantly faster but less precise
- Lemmatization improves TF-IDF and PCA results
- Bigrams enhance semantic representation
- The quality of preprocessing directly impacts downstream tasks

## Conclusions

- No single library solves the entire NLP pipeline
- CoreNLP is better for semantic quality and analysis tasks
- OpenNLP is better for performance-critical applications
- Custom preprocessing (cleaning, TF-IDF, PCA) is essential

## Limitations

- PCA is used as an approximation for topic modeling
- Performance depends on hardware resources