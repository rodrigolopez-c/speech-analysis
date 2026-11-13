# NLP Pipeline: CoreNLP vs OpenNLP & Módulos Avanzados

## Overview

Este proyecto implementa un pipeline completo de Procesamiento de Lenguaje Natural (NLP) híbrido (Java + Python) para analizar documentos textuales (informes institucionales, debates y corpus en español/inglés). 

El sistema combina análisis lingüístico clásico de alta precisión con modelos neuronales de última generación:

1. **Stanford CoreNLP & Apache OpenNLP**: Preprocesamiento lingüístico (Tokenización, POS Tagging, NER, Lemmatización, BoW, TF-IDF, PCA).
2. **Análisis de Emociones Concretas (NRC EmoLex & NRC Affect Intensity Lexicon)**: Sustituye la polaridad escalar simple por las 8 emociones básicas de Plutchik ponderadas sintácticamente con Stanford CoreNLP.
3. **Detección de Sarcasmo**: Arquitecturas Transformers de Hugging Face (BERT, RoBERTa, XLNet), detección de incongruencias semánticas locales (inspirado en DeepMoji) y modelos de línea base con Scikit-Learn (SVM, Regresión Logística).
4. **Detección de Metáforas**: Red Neuronal Recurrente Bidireccional (BiLSTM) alimentada por embeddings contextuales, enriquecida con índices de concreción/abstracción y conjuntos de sinónimos de WordNet (arquitecturas inspiradas en IlliniMet y MetaNet).

---

## Estructura del Pipeline NLP

```
Texto (Corpus CSV / Documentos)
  │
  ├──► Preprocesamiento Stanford CoreNLP / OpenNLP
  │      ├── Tokenización y Sentence Segmentation
  │      ├── POS Tagging (Etiquetado gramatical)
  │      └── Lemmatización y Limpieza de Texto
  │
  ├──► Análisis de Emociones Concretas (NRC EmoLex + Intensidad) [Java]
  │      ├── Mapeo de palabras a las 8 emociones de Plutchik
  │      ├── Ponderación gramatical mediante POS Tags (Adjetivos 1.5x, Verbos 1.2x, Sustantivos 1.0x)
  │      ├── Integración de Intensidad Afectiva (NRC Affective Intensity Lexicon)
  │      └── Construcción de Vectores de Características Emocionales
  │
  ├──► Vectorización Tradicional & Reducción
  │      ├── Bag of Words (BoW) & N-gramas
  │      ├── TF-IDF Vectorization
  │      └── PCA (Aproximación de tópicos y visualización 2D)
  │
  └──► Módulos Avanzados de Aprendizaje Profundo [Python Bridge]
         ├── Detección de Sarcasmo:
         │     ├── Transformers Hugging Face (BERT / RoBERTa / XLNet contextual)
         │     ├── Incongruencias Semánticas y Emocionales (DeepMoji)
         │     └── Línea Base con Scikit-Learn (SVM y Regresión Logística)
         └── Detección de Metáforas:
               ├── BiLSTM sobre Embeddings Contextuales (768-dim)
               ├── Enriquecimiento Lingüístico: Índices de Concreción / Abstracción
               ├── Dominios Semánticos y Synsets vía WordNet (NLTK OMW)
               └── Marcos Conceptuales (MetaNet: Dominio Fuente -> Dominio Destino)
```

---

## Estructura del Proyecto

```
nlp-pipeline/
├── pom.xml
├── readme.md
│
├── src/main/java/com/nlp/pipeline/
│   ├── model/
│   │   ├── DocumentResult.java         # Modelo de datos anotado
│   │   └── EmotionProfile.java         # Perfil de 8 emociones NRC + Intensidad
│   ├── nlp/
│   │   ├── CoreNLPService.java         # Servicio Stanford CoreNLP
│   │   ├── OpenNLPService.java         # Servicio Apache OpenNLP
│   │   ├── NRCEmotionService.java      # Servicio NRC EmoLex + POS Tags
│   │   ├── PythonPipelineBridge.java   # Puente de ejecución Java -> Python
│   │   ├── BigramGenerator.java
│   │   ├── SentimentService.java
│   │   └── TextNormalizer.java
│   ├── pipeline/
│   │   ├── CorpusManager.java
│   │   ├── CorpusSummary.java
│   │   ├── DocumentProcessor.java      # Orquestador de preprocesamiento
│   │   ├── DocumentExporter.java       # Exportador interoperable (CSV)
│   │   ├── SentimentSummary.java       # Reporte detallado de emociones NRC
│   │   └── ClassificationSummary.java
│   ├── vectorization/
│   │   ├── BagOfWords.java
│   │   ├── TFIDFVectorizer.java
│   │   ├── PCAReducer.java
│   │   └── VocabularyBuilder.java
│   └── App.java                        # Punto de entrada Java
│
├── src/main/resources/
│   ├── datasets/                       # Datasets de entrada (.csv, .xlsx)
│   ├── lexicons/
│   │   └── nrc_emolex.csv              # Léxico NRC bilingüe (11,800+ palabras)
│   ├── models/                         # Modelos OpenNLP (.bin)
│   └── stopwords.txt
│
└── src/python/
    ├── sarcasm/
    │   ├── baseline_detector.py        # Scikit-Learn SVM & Regresión Logística
    │   ├── transformer_detector.py     # Hugging Face Transformers (BERT, RoBERTa, XLNet)
    │   └── semantic_incongruity.py     # Incongruencia semántica y emocional (DeepMoji)
    ├── metaphor/
    │   ├── bilstm_model.py             # Red BiLSTM en PyTorch sobre embeddings
    │   ├── linguistic_features.py      # Concreción, abstracción y WordNet
    │   └── metaphor_detector.py        # Motor IlliniMet & MetaNet
    ├── pipeline_runner.py              # CLI para procesamiento de corpus y demos
    └── requirements.txt
```

---

## Instalación y Requisitos

### Java
- Java 21+ o 23
- Maven 3.9+

Compilar el proyecto:
```bash
mvn clean compile
```

### Python
- Python 3.10+
- Dependencias (PyTorch, Transformers, Scikit-Learn, NLTK):
```bash
pip install -r src/python/requirements.txt
```

---

## Ejecución

### 1. Ejecución del Pipeline en Java
Ejecuta el pipeline completo, procesando el corpus con Stanford CoreNLP, construyendo los vectores de emociones con NRC EmoLex y exportando los resultados:
```bash
mvn exec:java "-Dexec.mainClass=com.nlp.pipeline.App" "-Dexec.jvmArgs=-Xms2g -Xmx6g"
```

### 2. Ejecución Demostrativa de Sarcasmo y Metáforas (Python)
Para probar casos paradigmáticos interactivos de sarcasmo y metáforas:
```bash
python src/python/pipeline_runner.py --demo
```

### 3. Ejecución por Lotes sobre un Dataset
Para procesar un dataset de texto o el corpus exportado por Java:
```bash
python src/python/pipeline_runner.py --csv target/processed_informes_iteso_trafilatura.csv --sample 50
```

---

## Módulos Detallados

### 1. Análisis de Emociones Concretas (NRC EmoLex)
En lugar de una métrica simplista de polaridad (1 a 3), el sistema modela las 8 emociones básicas de Robert Plutchik:
- **Ira** (*Anger*)
- **Asco** (*Disgust*)
- **Miedo** (*Fear*)
- **Tristeza** (*Sadness*)
- **Sorpresa** (*Surprise*)
- **Anticipación** (*Anticipation*)
- **Alegría** (*Joy*)
- **Confianza** (*Trust*)

**Integración con StanfordNLP:** Cada palabra es analizada conjuntamente con su etiqueta POS (`CoreAnnotations.PartOfSpeechAnnotation`). Los adjetivos (`JJ`) y adverbios (`RB`) reciben una ponderación superior (1.5x y 1.3x) debido a su función intensificadora del afecto, mientras que los verbos (`VB`) se ponderan a 1.2x y los sustantivos (`NN`) a 1.0x. Adicionalmente, se integra el puntaje continuo de intensidad afectiva del **NRC Affective Intensity Lexicon**.

### 2. Detección de Sarcasmo
- **Modelos de Lenguaje Contextuales (Hugging Face Transformers)**: Permite utilizar arquitecturas preentrenadas como BERT (`dccuchile/bert-base-spanish-wwm-cased`), RoBERTa (`cardiffnlp/twitter-roberta-base-irony`) y XLNet considerando ventanas de contexto conversacional previo.
- **Incongruencia Semántica y Emocional (DeepMoji)**: Detecta la presencia de valencia verbal superficial fuertemente positiva en conjunción con situaciones o palabras de afecto negativo intrínseco.
- **Línea Base (Scikit-Learn)**: Implementa clasificadores lineales SVM (`LinearSVC`) y Regresión Logística (`LogisticRegression`) con unión de características TF-IDF n-gramas a nivel de palabra y carácter, marcadores ortográficos y rasgos de contraste.

### 3. Detección de Metáforas
- **Red Neuronal BiLSTM**: Red recurrente bidireccional en PyTorch que procesa la secuencia de tokens enriquecida con la concatenación de embeddings contextuales y vectores lingüísticos.
- **Índices de Concreción y Abstracción**: Basado en normas psicosemánticas y sufijación derivativa para evaluar saltos de concreción (1.0 = abstracto a 5.0 = concreto).
- **WordNet & Dominios Semánticos (IlliniMet)**: Consulta los synsets y supersenses (`lexnames`) de NLTK WordNet (con soporte en español vía Open Multilingual WordNet) para calcular disparidad de dominios semánticos.
- **Semántica de Marcos (MetaNet)**: Identifica proyecciones metafóricas conceptuales paradigmáticas entre Dominios Fuente (físicos/biológicos) y Dominios Destino (sociales/institucionales).