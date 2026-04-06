package com.nlp.pipeline.pipeline;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.nlp.pipeline.model.DocumentResult;
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

        try {
            Files.list(Paths.get(basePath))
                    .filter(Files::isDirectory)
                    .forEach(categoryPath -> {

                        String category = categoryPath.getFileName().toString();
                        int count = 0;

                        try (java.util.stream.Stream<Path> stream = Files.list(categoryPath)) {
                            List<Path> files = stream.collect(java.util.stream.Collectors.toList());
                            for (Path file : files) {

                                if (limitPerCategory > 0 && count >= limitPerCategory) break;

                                if (Files.isRegularFile(file)) {
                                    String content = FileUtils.readFile(file);

                                    documents.add(new DocumentResult(
                                            category,
                                            file.getFileName().toString(),
                                            content
                                    ));

                                    count++;
                                }
                            }
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    });

        } catch (IOException e) {
            throw new RuntimeException("Error cargando corpus", e);
        }

        return documents;
    }
}