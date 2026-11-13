package com.nlp.pipeline.pipeline;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import com.nlp.pipeline.model.DocumentResult;
import com.nlp.pipeline.utils.CSVDatasetLoader;
import com.nlp.pipeline.utils.ExcelDatasetLoader;
import com.nlp.pipeline.utils.FileUtils;

public class CorpusManager {

    private final String basePath;
    private final int limitPerCategory;

    public CorpusManager(String basePath, int limitPerCategory) {
        this.basePath = basePath;
        this.limitPerCategory = limitPerCategory;
    }

    public List<DocumentResult> loadCorpus() {
        List<DocumentResult> documents = new ArrayList<>();
        Path rootPath = Paths.get(basePath);

        if (!Files.exists(rootPath)) {
            Path fallbackPath = Paths.get("target/classes/datasets");
            if (Files.exists(fallbackPath)) {
                rootPath = fallbackPath;
            } else {
                System.err.println("El directorio/archivo especificado no existe: " + rootPath.toAbsolutePath());
                return documents;
            }
        }

        System.out.println("Ruta absoluta del corpus: " + rootPath.toAbsolutePath());

        // Si la ruta dada es directamente un archivo CSV o Excel
        if (Files.isRegularFile(rootPath)) {
            String fileName = rootPath.toString().toLowerCase();
            if (fileName.endsWith(".csv")) {
                return CSVDatasetLoader.loadCSVDataset(rootPath.toFile(), limitPerCategory);
            } else if (fileName.endsWith(".xlsx") || fileName.endsWith(".xls")) {
                return ExcelDatasetLoader.loadExcelDataset(rootPath.toFile(), limitPerCategory);
            }
        }

        try {
            // Buscar archivos CSV en la carpeta basePath
            try (Stream<Path> csvStream = Files.list(rootPath)) {
                List<Path> csvFiles = csvStream
                        .filter(Files::isRegularFile)
                        .filter(p -> p.toString().toLowerCase().endsWith(".csv"))
                        .toList();

                for (Path csvFile : csvFiles) {
                    System.out.println("Cargando dataset CSV: " + csvFile.getFileName());
                    documents.addAll(CSVDatasetLoader.loadCSVDataset(csvFile.toFile(), limitPerCategory));
                }
            }

            // Buscar archivos Excel en la carpeta basePath
            try (Stream<Path> excelStream = Files.list(rootPath)) {
                List<Path> excelFiles = excelStream
                        .filter(Files::isRegularFile)
                        .filter(p -> p.toString().toLowerCase().endsWith(".xlsx") || p.toString().toLowerCase().endsWith(".xls"))
                        .toList();

                for (Path excelFile : excelFiles) {
                    System.out.println("Cargando dataset Excel: " + excelFile.getFileName());
                    documents.addAll(ExcelDatasetLoader.loadExcelDataset(excelFile.toFile(), limitPerCategory));
                }
            }

            // Buscar subdirectorios (estructura de carpetas por categoría)
            try (Stream<Path> dirStream = Files.list(rootPath)) {
                dirStream.filter(Files::isDirectory)
                        .forEach(categoryPath -> {
                            String category = categoryPath.getFileName().toString();
                            int count = 0;

                            try (Stream<Path> stream = Files.list(categoryPath)) {
                                List<Path> files = stream.toList();
                                for (Path file : files) {
                                    if (limitPerCategory > 0 && count >= limitPerCategory) break;

                                    if (Files.isRegularFile(file) && !file.getFileName().toString().startsWith(".")) {
                                        String content = FileUtils.readFile(file);
                                        if (content != null && !content.trim().isEmpty()) {
                                            documents.add(new DocumentResult(
                                                    category,
                                                    file.getFileName().toString(),
                                                    content
                                            ));
                                            count++;
                                        }
                                    }
                                }
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }
                        });
            }

        } catch (IOException e) {
            throw new RuntimeException("Error cargando corpus desde: " + basePath, e);
        }

        return documents;
    }
}