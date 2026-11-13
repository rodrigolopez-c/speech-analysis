package com.nlp.pipeline.utils;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nlp.pipeline.model.DocumentResult;

public class CSVDatasetLoader {

    public static List<DocumentResult> loadCSVDataset(File csvFile, int limitPerCategory) {
        List<DocumentResult> documents = new ArrayList<>();
        Map<String, Integer> categoryCount = new HashMap<>();

        if (!csvFile.exists()) {
            System.err.println("Archivo CSV no existe: " + csvFile.getAbsolutePath());
            return documents;
        }

        try {
            String content = readRawFileWithEncoding(csvFile);
            if (content.isEmpty()) return documents;

            // Detectar delimitador leyendo la primera línea
            char delimiter = ',';
            int firstNL = content.indexOf('\n');
            String headerSnippet = firstNL != -1 ? content.substring(0, firstNL) : content;
            int semicolons = 0, commas = 0, tabs = 0;
            for (char c : headerSnippet.toCharArray()) {
                if (c == ';') semicolons++;
                else if (c == ',') commas++;
                else if (c == '\t') tabs++;
            }
            if (semicolons > commas && semicolons > tabs) delimiter = ';';
            else if (tabs > commas && tabs > semicolons) delimiter = '\t';

            System.out.println("CSV Delimitador detectado: '" + (delimiter == '\t' ? "\\t" : delimiter) + "' para " + csvFile.getName());

            List<List<String>> records = parseCSVRecords(content, delimiter);
            if (records.isEmpty()) return documents;

            List<String> header = records.get(0);
            int speakerCol = -1;
            int textCol = -1;
            int idCol = -1;
            int dateCol = -1;

            for (int i = 0; i < header.size(); i++) {
                String val = header.get(i).trim().toLowerCase();
                if (val.contains("speaker") || val.contains("author") || val.contains("autor") || val.contains("category") || val.contains("president")) {
                    if (speakerCol == -1 || val.equals("president") || val.equals("speaker")) {
                        speakerCol = i;
                    }
                } else if (val.equals("text_clean") || val.contains("text-original") || val.equals("text") || val.contains("content") || val.equals("speech")) {
                    if (textCol == -1 || val.equals("text_clean")) {
                        textCol = i;
                    }
                } else if (val.equals("id") || val.equals("title")) {
                    idCol = i;
                } else if (val.contains("date") || val.contains("fecha")) {
                    dateCol = i;
                }
            }

            if (textCol == -1) textCol = header.size() > 6 ? 6 : header.size() - 1;
            if (speakerCol == -1) speakerCol = header.size() > 2 ? 2 : 0;

            System.out.println("CSV Columnas detectadas -> Speaker/President: " + speakerCol + " (" + (speakerCol < header.size() ? header.get(speakerCol) : "N/A") + "), Text/Speech: " + textCol + " (" + (textCol < header.size() ? header.get(textCol) : "N/A") + "), ID/Title: " + (idCol != -1 && idCol < header.size() ? idCol + " (" + header.get(idCol) + ")" : "N/A") + ", Date: " + (dateCol != -1 && dateCol < header.size() ? dateCol + " (" + header.get(dateCol) + ")" : "N/A"));

            int rowIdx = 1;
            for (int r = 1; r < records.size(); r++) {
                List<String> record = records.get(r);
                if (record.isEmpty() || (record.size() == 1 && record.get(0).isEmpty())) continue;

                String speaker = (speakerCol < record.size()) ? cleanQuotes(record.get(speakerCol)) : "Desconocido";
                String text = "";
                if (textCol < record.size()) {
                    if (textCol == header.size() - 1 && record.size() > header.size()) {
                        List<String> remaining = record.subList(textCol, record.size());
                        text = cleanQuotes(String.join(String.valueOf(delimiter), remaining));
                    } else {
                        text = cleanQuotes(record.get(textCol));
                    }
                }
                String id = (idCol != -1 && idCol < record.size()) ? cleanQuotes(record.get(idCol)) : String.valueOf(rowIdx);
                String date = (dateCol != -1 && dateCol < record.size()) ? cleanQuotes(record.get(dateCol)) : "";

                rowIdx++;

                if (text.isEmpty() || text.length() < 10) continue;
                if (speaker.isEmpty()) speaker = "Desconocido";

                if (limitPerCategory > 0) {
                    int current = categoryCount.getOrDefault(speaker, 0);
                    if (current >= limitPerCategory) continue;
                    categoryCount.put(speaker, current + 1);
                }

                String docName = "doc_" + id.replaceAll("[^a-zA-Z0-9]", "_") + (date.isEmpty() ? "" : "_" + date.replaceAll("[^a-zA-Z0-9]", "_")) + ".txt";
                documents.add(new DocumentResult(speaker, docName, text));
            }

            System.out.println("Total documentos cargados desde CSV (" + csvFile.getName() + "): " + documents.size());

        } catch (Exception e) {
            System.err.println("Error leyendo CSV: " + csvFile.getAbsolutePath() + " - " + e.getMessage());
            e.printStackTrace();
        }

        return documents;
    }

    private static String readRawFileWithEncoding(File csvFile) {
        try {
            byte[] bytes = Files.readAllBytes(csvFile.toPath());
            CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder();
            decoder.onMalformedInput(CodingErrorAction.REPORT);
            decoder.onUnmappableCharacter(CodingErrorAction.REPORT);
            try {
                CharBuffer cb = decoder.decode(ByteBuffer.wrap(bytes));
                String str = cb.toString();
                if (str.startsWith("\uFEFF")) {
                    str = str.substring(1);
                }
                return str;
            } catch (CharacterCodingException e) {
                // Fallback to Latin-1
            }
            String latin1 = new String(bytes, StandardCharsets.ISO_8859_1);
            if (latin1.startsWith("\uFEFF")) {
                latin1 = latin1.substring(1);
            }
            return latin1;
        } catch (Exception e) {
            System.err.println("Error leyendo archivo: " + e.getMessage());
            return "";
        }
    }

    private static List<List<String>> parseCSVRecords(String content, char delimiter) {
        List<List<String>> records = new ArrayList<>();
        List<String> currentRecord = new ArrayList<>();
        StringBuilder currentCell = new StringBuilder();
        boolean inQuotes = false;
        int len = content.length();

        for (int i = 0; i < len; i++) {
            char c = content.charAt(i);

            if (delimiter == ';' && (c == '\r' || c == '\n')) {
                inQuotes = false;
            }

            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < len && content.charAt(i + 1) == '"') {
                        currentCell.append('"');
                        i++; // skip escaped quote
                    } else {
                        inQuotes = false;
                    }
                } else {
                    currentCell.append(c);
                }
            } else {
                if (c == '"') {
                    inQuotes = true;
                } else if (c == delimiter) {
                    currentRecord.add(currentCell.toString().trim());
                    currentCell.setLength(0);
                } else if (c == '\r') {
                    if (i + 1 < len && content.charAt(i + 1) == '\n') {
                        i++;
                    }
                    currentRecord.add(currentCell.toString().trim());
                    currentCell.setLength(0);
                    if (!currentRecord.isEmpty()) {
                        records.add(currentRecord);
                        currentRecord = new ArrayList<>();
                    }
                } else if (c == '\n') {
                    currentRecord.add(currentCell.toString().trim());
                    currentCell.setLength(0);
                    if (!currentRecord.isEmpty()) {
                        records.add(currentRecord);
                        currentRecord = new ArrayList<>();
                    }
                } else {
                    currentCell.append(c);
                }
            }
        }

        if (currentCell.length() > 0 || !currentRecord.isEmpty()) {
            currentRecord.add(currentCell.toString().trim());
            records.add(currentRecord);
        }

        return records;
    }

    private static String cleanQuotes(String val) {
        if (val == null) return "";
        val = val.trim();
        if ((val.startsWith("'") && val.endsWith("'")) || (val.startsWith("\"") && val.endsWith("\""))) {
            if (val.length() >= 2) {
                val = val.substring(1, val.length() - 1).trim();
            }
        }
        return val;
    }
}



