# NLP Pipeline: Análisis de Sentimiento (NRC Plutchik), Sarcasmo y Metáforas

## Descripción General

Este proyecto implementa un pipeline híbrido (**Java + Python**) de Procesamiento de Lenguaje Natural (NLP) diseñado para analizar discursos políticos, debates parlamentarios e informes institucionales en español e inglés.

El sistema combina análisis lingüístico clásico de alta precisión con modelos de aprendizaje automático y profundo:

1. **Stanford CoreNLP & Apache OpenNLP (Java)**: Preprocesamiento lingüístico (Tokenización, Segmentación de oraciones, POS Tagging, NER, Lematización), vectorización (`Bag of Words`, `TF-IDF`), reducción de dimensionalidad (`PCA`) y clasificación de autor/categoría.
2. **Análisis de Emociones Concretas (NRC EmoLex + Intensidad Afectiva) (Java)**: Sustituye la polaridad unidimensional por el modelo de las **8 emociones básicas de Robert Plutchik** (*Confianza, Anticipación, Alegría, Miedo, Tristeza, Sorpresa, Asco, Ira*), ponderadas sintácticamente según la categoría gramatical (`POS Tag`) y un léxico bilingüe curado de **12,115 términos**.
3. **Detección de Sarcasmo (Python)**: Clasificadores de línea base con Scikit-Learn (`SVM` y `Regresión Logística` sobre n-gramas de palabras/caracteres y marcadores pragmáticos), arquitecturas Transformers (`BERT`, `RoBERTa`, `XLNet`) y cálculo de incongruencia semántica/emocional local estilo `DeepMoji`.
4. **Detección de Metáforas (Python)**: Red Neuronal Recurrente Bidireccional (`BiLSTM` en PyTorch) combinada con índices psicosemánticos de concreción/abstracción, disparidad de dominios en `WordNet` (`Open Multilingual WordNet`) e identificación de marcos conceptuales de `MetaNet` (`CONSTRUCCIÓN_ES_DESARROLLO`, `VIDA_ES_UN_VIAJE`, `DEBATE_ES_GUERRA`, `INSTITUCIÓN_ES_ORGANISMO`).

---

## Arquitectura del Flujo de Trabajo

```
Archivo de Entrada (.csv / .xlsx / carpeta de textos)
  │
  ├──► [PASO 1: Preprocesamiento en Java — Stanford CoreNLP & OpenNLP]
  │      ├── Carga de corpus (CorpusManager)
  │      ├── Tokenización, Lematización, POS Tagging y NER (DocumentProcessor)
  │      └── Vectorización BoW, TF-IDF, reducción PCA (2D) y Clasificación de Autor
  │
  ├──► [PASO 2: Análisis Afectivo Multidimensional en Java — NRC EmoLex]
  │      ├── Mapeo léxico a las 8 emociones de Plutchik (12,115 términos ES/EN)
  │      ├── Ponderación gramatical POS (Adjetivos 1.5x, Adverbios 1.3x, Verbos 1.2x, Sustantivos 1.0x)
  │      ├── Integración continua de intensidad (NRC Affective Intensity Lexicon)
  │      └── Exportación interoperable a CSV -> target/processed_<dataset>.csv
  │
  └──► [PASO 3: Módulos Avanzados en Python — PythonPipelineBridge]
         ├── Detección de Sarcasmo:
         │     ├── Propensión al sarcasmo con SVM / Regresión Logística (umbral >= 0.60)
         │     ├── Incongruencia semántica y emocional local (DeepMoji)
         │     └── Soporte para Transformers (BERT, RoBERTa, XLNet)
         └── Detección de Metáforas:
               ├── Clasificación secuencial con BiLSTM en PyTorch (umbral >= 0.35)
               ├── Índices de concreción/abstracción + WordNet Supersenses (IlliniMet)
               ├── Identificación de marcos conceptuales (MetaNet)
               └── Exportación de resultados -> target/advanced_nlp_results.csv
```

---

## Estructura del Proyecto

```
speech-analysis/
├── pom.xml                                 # Configuración Maven y dependencias Java
├── readme.md                               # Guía paso a paso del proyecto
│
├── docs/                                   # Reportes de resultados y recursos visuales
│   ├── informe_nlp_pipeline.md                         # Resultados: Informes del Rector ITESO
│   ├── informe_nlp_pipeline_diario_debates.md          # Resultados: Congreso de Jalisco
│   ├── informe_nlp_pipeline_diario_debates_senado.md   # Resultados: Senado de la República
│   ├── informe_nlp_pipeline_presidential_speeches.md   # Resultados: Discursos Presidenciales EE. UU.
│   ├── recursos_visuales_poster.md                     # Tablas, diagramas y gráficas para póster
│   └── figuras_poster/                                 # Gráficas PNG en alta resolución (300 DPI)
│
├── src/main/java/com/nlp/pipeline/
│   ├── App.java                            # Punto de entrada principal del pipeline
│   ├── model/
│   │   ├── DocumentResult.java             # Modelo de documento anotado
│   │   └── EmotionProfile.java             # Perfil de las 8 emociones NRC + intensidad
│   ├── nlp/
│   │   ├── CoreNLPService.java             # Integración con Stanford CoreNLP
│   │   ├── OpenNLPService.java             # Integración con Apache OpenNLP
│   │   ├── NRCEmotionService.java          # Cálculo de emociones de Plutchik + POS weighting
│   │   ├── PythonPipelineBridge.java       # Puente de ejecución automática Java -> Python
│   │   ├── BigramGenerator.java
│   │   ├── SentimentService.java
│   │   └── TextNormalizer.java
│   ├── pipeline/
│   │   ├── CorpusManager.java              # Lector multi-formato (.csv, .xlsx, directorios)
│   │   ├── CorpusSummary.java
│   │   ├── DocumentProcessor.java          # Orquestador de preprocesamiento
│   │   ├── DocumentExporter.java           # Exportador de resultados de Java a CSV
│   │   ├── SentimentSummary.java           # Reporte estadístico de emociones en consola
│   │   └── ClassificationSummary.java
│   ├── utils/
│   │   ├── CSVDatasetLoader.java
│   │   ├── ExcelDatasetLoader.java
│   │   └── FileUtils.java
│   └── vectorization/
│       ├── BagOfWords.java
│       ├── TFIDFVectorizer.java
│       ├── PCAReducer.java
│       └── VocabularyBuilder.java
│
├── src/main/resources/                     # Carpeta central de recursos del pipeline
│   ├── datasets/                           # Datasets del proyecto (comprimidos en .zip y .csv activos)
│   │   ├── diario_debates_senado.zip       # Senado de la República (descomprimir a .csv para usar)
│   │   ├── diario_debates.zip              # Congreso de Jalisco (descomprimir a .csv para usar)
│   │   ├── informes_iteso.zip              # Informes del Rector ITESO (descomprimir a .csv para usar)
│   │   └── presidential_speeches.zip       # Discursos Presidenciales EE. UU. (descomprimir a .csv para usar)
│   ├── lexicons/
│   │   ├── nrc_emolex.csv                  # Léxico NRC bilingüe activo (12,115 términos)
│   │   └── nrc_emolex_es.csv               # Léxico NRC base en español
│   ├── models/                             # Modelos binarios preentrenados de OpenNLP (.bin)
│   │   ├── en-chunker.bin
│   │   ├── en-parser-chunking.bin
│   │   ├── en-pos-maxent.bin
│   │   ├── en-sent.bin
│   │   └── en-token.bin
│   ├── stopwords.txt                       # Lista de palabras vacías filtradas por TextNormalizer
│   └── training.txt                        # Archivo generado por OpenNLPService para clasificar autor
│
└── src/python/
    ├── sarcasm/
    │   ├── baseline_detector.py            # Scikit-Learn SVM & Regresión Logística
    │   ├── transformer_detector.py         # Hugging Face Transformers (BERT, RoBERTa, XLNet)
    │   └── semantic_incongruity.py         # Detector de incongruencia semántica (DeepMoji)
    ├── metaphor/
    │   ├── bilstm_model.py                 # Red BiLSTM en PyTorch sobre embeddings
    │   ├── linguistic_features.py          # Concreción/abstracción y WordNet (NLTK OMW)
    │   └── metaphor_detector.py            # Motor integrado tipo IlliniMet & MetaNet
    ├── pipeline_runner.py                  # CLI de Python para ejecución por lotes o demo
    └── requirements.txt                    # Dependencias de Python
```

---

## Funcionamiento Basado en la Carpeta `src/main/resources/`

Todo el comportamiento del pipeline (tanto el motor en Java como los módulos posteriores en Python) **opera en función de los archivos contenidos en `src/main/resources/`**. A continuación se detalla el rol de cada subcarpeta y archivo:

### 1. Gestión de Corpus: `datasets/` vs. `unused_datasets/`
- **`src/main/resources/datasets/` (Carpeta de trabajo activa):**
  Es el directorio de entrada principal que consume `CorpusManager`. Si se le pasa la ruta del directorio `src/main/resources/datasets` (o la ruta de un archivo dentro de él en `App.java`), el script carga automáticamente todos los archivos `.csv`, `.xlsx`/`.xls` o subcarpetas de texto que se encuentren aquí.
- **`src/main/resources/unused_datasets/` (Carpeta de reserva):**
  Contiene los demás corpus del proyecto (`informes_iteso.csv`, `diario_debates.csv`, `presidential_speeches.csv`) para evitar que se procesen todos al mismo tiempo si no se requieren en la corrida actual.
- **¿Cómo cambiar el corpus a analizar?**
  1. Mueve el archivo que deseas analizar desde `src/main/resources/unused_datasets/` hacia `src/main/resources/datasets/` (y mueve a `unused_datasets/` los que no quieras incluir en esa ejecución).
  2. Indica la ruta en `App.java` (o pásala por línea de comandos con `-Dexec.args="src/main/resources/datasets"` para que procese todo lo que esté dentro de `datasets/`).

### 2. Léxicos de Emociones (`src/main/resources/lexicons/`)
- **`nrc_emolex.csv`**: Archivo maestro cargado automáticamente por `NRCEmotionService` al iniciar el pipeline. Contiene las **12,115 palabras** (español e inglés) asociadas a las 8 emociones de Plutchik (`anger`, `disgust`, `fear`, `sadness`, `surprise`, `anticipation`, `joy`, `trust`), su polaridad y su puntaje continuo de intensidad (`intensity`).
- **`nrc_emolex_es.csv`**: Léxico de respaldo en español. Si deseas agregar nuevos términos o ajustar pesos léxicos, basta con editar `nrc_emolex.csv` respetando sus columnas.

### 3. Modelos Lingüísticos Binarios (`src/main/resources/models/`)
- Contiene los modelos preentrenados de Apache OpenNLP (`en-sent.bin`, `en-token.bin`, `en-pos-maxent.bin`, `en-chunker.bin`, `en-parser-chunking.bin`) empleados durante la segmentación, tokenización y etiquetado gramatical.

### 4. Normalización y Entrenamiento (`stopwords.txt` y `training.txt`)
- **`src/main/resources/stopwords.txt`**: Lista de palabras vacías que `TextNormalizer` lee automáticamente para limpiar cada documento antes de construir la matriz `Bag of Words`, `TF-IDF` y el vocabulario.
- **`src/main/resources/training.txt`**: Archivo de entrenamiento autogenerado en cada ejecución por `OpenNLPService.trainModel()`. El script toma los textos procesados y sus etiquetas de autor/orador, escribe este archivo en formato `<Categoria>\t<Texto>` y entrena en caliente el modelo clasificador de autor (`DocumentCategorizerME`).

---

## Guía de Instalación Paso a Paso

### Paso 1: Verificar requisitos del sistema
Asegúrate de contar con las siguientes herramientas instaladas y disponibles en tu variable de entorno `PATH`:
- **Java JDK**: versión **23** (o JDK 21+ ajustando `<maven.compiler.source>` y `<maven.compiler.target>` en `pom.xml`).
- **Apache Maven**: versión **3.9+**.
- **Python**: versión **3.10+**.

Puedes verificar tus versiones ejecutando en la terminal:
```powershell
java -version
mvn -version
python --version
```

---

### Paso 2: Instalar las dependencias de Python
Desde la raíz del proyecto (`speech-analysis`), instala los paquetes requeridos (`torch`, `transformers`, `scikit-learn`, `nltk`, `pandas`, `numpy`):

```powershell
pip install -r src/python/requirements.txt
```

*(Opcional pero recomendado)* Descarga los recursos léxicos de `WordNet` y `Open Multilingual WordNet` en `NLTK` para el análisis de dominios semánticos en español e inglés:

```powershell
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

### Paso 3: Compilar el proyecto Java con Maven
Compila el código fuente de Java y descarga automáticamente las dependencias de **Stanford CoreNLP**, **Apache OpenNLP**, **Commons Math** y **Apache POI**:

```powershell
mvn clean compile
```

---

## Guía de Uso Paso a Paso

Puedes ejecutar el proyecto de **tres maneras**: el flujo completo automatizado desde Java (que invoca a Python al finalizar), la prueba interactiva rápida en Python, o el análisis por lotes en Python sobre un CSV específico.

---

### Opción A: Ejecución Completa del Pipeline (Java + Python Bridge)

Al ejecutar `com.nlp.pipeline.App`, el sistema realiza todo el flujo de extremo a extremo:
1. Carga el dataset desde `src/main/resources/datasets/` (o la ruta que le indiques).
2. Ejecuta tokenización, POS Tagging, lematización, BoW, TF-IDF, PCA y clasificación de autor en Java.
3. Calcula el perfil de las **8 emociones de Plutchik (NRC EmoLex)** e imprime las tablas de resumen por corpus y por autor/orador en consola.
4. Exporta el corpus anotado a `target/processed_<nombre_dataset>.csv`.
5. Invoca automáticamente `src/python/pipeline_runner.py` mediante `PythonPipelineBridge` para evaluar **Sarcasmo** y **Metáforas** sobre todos los documentos procesados.

#### 1. Ejecutar con el dataset configurado por defecto en `src/main/resources/datasets/`
Por defecto, `App.java` toma el archivo activo ubicado en `src/main/resources/datasets/diario_debates_senado_trafilatura.csv` y utiliza automáticamente los léxicos (`src/main/resources/lexicons/nrc_emolex.csv`), modelos (`src/main/resources/models/`) y `stopwords.txt` de la carpeta de recursos:

```powershell
mvn exec:java "-Dexec.mainClass=com.nlp.pipeline.App" "-Dexec.jvmArgs=-Xms2g -Xmx6g"
```

#### 2. Procesar todos los archivos que coloques dentro de `src/main/resources/datasets/`
Si mueves uno o varios datasets desde `src/main/resources/unused_datasets/` (como `informes_iteso.csv`, `diario_debates.csv` o `presidential_speeches.csv`) hacia `src/main/resources/datasets/`, puedes indicarle al script que procese la carpeta completa `src/main/resources/datasets`:

```powershell
mvn exec:java "-Dexec.mainClass=com.nlp.pipeline.App" "-Dexec.jvmArgs=-Xms2g -Xmx6g" "-Dexec.args=src/main/resources/datasets"
```

#### 3. Ejecutar especificando la ruta de uno o varios datasets en `src/main/resources/`
También puedes apuntar directamente a cualquier archivo dentro de `datasets/` o `unused_datasets/` mediante `-Dexec.args`:

```powershell
mvn exec:java "-Dexec.mainClass=com.nlp.pipeline.App" "-Dexec.jvmArgs=-Xms2g -Xmx6g" "-Dexec.args=src/main/resources/unused_datasets/informes_iteso.csv"
```

Para procesar múltiples archivos específicos en una sola corrida, sepáralos por un espacio dentro de `-Dexec.args`:

```powershell
mvn exec:java "-Dexec.mainClass=com.nlp.pipeline.App" "-Dexec.jvmArgs=-Xms2g -Xmx6g" "-Dexec.args=src/main/resources/unused_datasets/informes_iteso.csv src/main/resources/unused_datasets/diario_debates.csv"
```

---

### Opción B: Ejecución Demostrativa Rápida en Python (`--demo`)

Si deseas probar inmediatamente los detectores de **Sarcasmo** (`SVM` + `DeepMoji`) y **Metáforas** (`BiLSTM` + `WordNet` + `MetaNet`) con oraciones de ejemplo (metafóricas, sarcásticas y literales) sin procesar un corpus completo:

```powershell
python src/python/pipeline_runner.py --demo
```

**Salida esperada en consola:**
- Para cada oración de prueba, verás:
  - `[Sarcasmo]`: Probabilidad `SVM`, puntaje de incongruencia `DeepMoji` y etiqueta (`SARCÁSTICO` / `LITERAL`).
  - `[Metáfora]`: Puntuación de metaforicidad, tokens metafóricos detectados, marcos conceptuales activos (`CONSTRUCCIÓN_ES_DESARROLLO`, `INSTITUCIÓN_ES_ORGANISMO`, etc.) y etiqueta (`METAFÓRICO` / `LITERAL`).

---

### Opción C: Ejecución Independiente del Módulo Python sobre un CSV (`--csv`)

Si ya tienes un archivo CSV (por ejemplo, uno generado previamente en `target/processed_*.csv` o cualquier CSV externo) y quieres ejecutar únicamente la detección de **Sarcasmo y Metáforas**:

#### 1. Analizar una submuestra rápida (por ejemplo, los primeros 50 documentos)
```powershell
python src/python/pipeline_runner.py --csv src/main/resources/datasets/diario_debates_senado_trafilatura.csv --sample 50
```

#### 2. Analizar el corpus completo (`--sample -1`) y elegir ruta de salida (`--output`)
```powershell
python src/python/pipeline_runner.py --csv target/processed_diario_debates_senado_trafilatura.csv --sample -1 --output target/resultados_senado_completos.csv
```

#### Parámetros disponibles en `pipeline_runner.py`:
| Argumento | Tipo | Valor por defecto | Descripción |
| :--- | :---: | :---: | :--- |
| `--demo` | Flag | `False` | Ejecuta la demostración interactiva con oraciones paradigmáticas. |
| `--csv` | `str` | `None` | Ruta al archivo CSV de entrada a analizar. |
| `--output` | `str` | `target/advanced_nlp_results.csv` | Ruta donde se guardará el CSV enriquecido con las nuevas columnas. |
| `--sample` | `int` | `50` | Número de filas a evaluar. Usa `-1` para procesar **todo** el dataset. |

---

## Cómo Agregar y Analizar tu Propio Dataset

El lector `CorpusManager` y el script `pipeline_runner.py` aceptan:
1. **Archivos CSV (`.csv`)** delimitados por coma (`,`) o punto y coma (`;`).
2. **Archivos Excel (`.xlsx` / `.xls`)**.
3. **Directorios categorizados**, donde cada subcarpeta representa una categoría/autor y contiene archivos `.txt`.

### Columnas reconocidas automáticamente
En archivos CSV/Excel, el sistema detecta automáticamente cualquiera de las siguientes columnas para el cuerpo del discurso:
- **Texto del documento**: `sampleText`, `text_clean`, `text-original`, `content`, `tokens`, `texto` (o la última columna disponible).
- **Categoría / Orador / Autor**: `category`, `speaker`, `author`, `presidente`, `rector`, `orador` (utilizada para agrupar estadísticas de sentimiento y entrenar el clasificador de autor).

---

## Archivos de Salida e Interpretación de Métricas

Al finalizar la ejecución, encontrarás dos archivos generados dentro de la carpeta `target/`:

1. **`target/processed_<dataset>.csv`** *(Generado por Java)*:
   - Contiene los textos limpios, tokens normalizados, conteos léxicos, puntajes ponderados para cada una de las **8 emociones de Plutchik** (`anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `trust`), la **emoción dominante** y la **intensidad afectiva media** (`0.0` a `1.0`).

2. **`target/advanced_nlp_results.csv`** *(Generado por Python)*:
   - Agrega al CSV anterior las siguientes 5 columnas analíticas:
     - `sarcasm_score_svm`: Propensión al sarcasmo (`0.0` a `1.0`). Valores **$\ge 0.60$** indican alto potencial sarcástico.
     - `sarcasm_incongruity`: Índice de tensión afectiva / incongruencia semántica local estilo `DeepMoji` (`0.0` a `1.0`).
     - `metaphor_score`: Puntuación de metaforicidad (`0.0` a `1.0`). Valores **$\ge 0.35$** indican presencia metafórica significativa.
     - `metaphor_tokens`: Hasta 5 palabras clave que operan en sentido figurado dentro del documento.
     - `conceptual_frames`: Marcos conceptuales de `MetaNet` identificados en el documento (`CONSTRUCCIÓN_ES_DESARROLLO`, `VIDA_ES_UN_VIAJE`, `DEBATE_ES_GUERRA`, `INSTITUCIÓN_ES_ORGANISMO` o `ninguno`).

---

## Reportes de Resultados y Recursos Visuales Incluidos

En el directorio `docs/` se encuentran documentados los resultados completos de las ejecuciones sobre los 4 corpus del proyecto (`2,450` documentos en total):

- [docs/informe_nlp_pipeline.md](docs/informe_nlp_pipeline.md): Resultados en los **Informes del Rector del ITESO** ($n = 32$).
- [docs/informe_nlp_pipeline_diario_debates.md](docs/informe_nlp_pipeline_diario_debates.md): Resultados en el **Diario de los Debates del Congreso de Jalisco** ($n = 694$).
- [docs/informe_nlp_pipeline_presidential_speeches.md](docs/informe_nlp_pipeline_presidential_speeches.md): Resultados en los **Discursos Presidenciales de Estados Unidos** ($n = 995$).
- [docs/informe_nlp_pipeline_diario_debates_senado.md](docs/informe_nlp_pipeline_diario_debates_senado.md): Resultados en el **Diario de los Debates del Senado de la República** ($n = 729$).
- [docs/recursos_visuales_poster.md](docs/recursos_visuales_poster.md): Síntesis comparativa trans-corpus, diagramas de arquitectura, tablas maestras y gráficas en alta resolución (`docs/figuras_poster/`) para presentación en póster científico.

---

## Solución de Problemas Comunes (Troubleshooting)

- **`OutOfMemoryError: Java heap space` durante el análisis de corpus grandes:**
  Asegúrate de incluir el argumento `-Dexec.jvmArgs="-Xms2g -Xmx6g"` al ejecutar Maven para asignar suficiente memoria a los modelos de Stanford CoreNLP.
- **Error de versión de Java (`release version 23 not supported`):**
  Si tienes instalado Java 21 en lugar de Java 23, edita las líneas `<maven.compiler.source>23</maven.compiler.source>` y `<maven.compiler.target>23</maven.compiler.target>` en `pom.xml` cambiándolas a `21`, y vuelve a ejecutar `mvn clean compile`.
- **`PythonPipelineBridge` no encuentra los módulos de Python:**
  Verifica que el comando `python` en tu terminal corresponda al entorno donde ejecutaste `pip install -r src/python/requirements.txt`.
