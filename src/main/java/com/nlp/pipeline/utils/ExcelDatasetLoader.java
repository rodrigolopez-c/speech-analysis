package com.nlp.pipeline.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.WorkbookFactory;

import com.nlp.pipeline.model.DocumentResult;

public class ExcelDatasetLoader {

    public static List<DocumentResult> loadExcelDataset(File excelFile, int limitPerCategory) {
        List<DocumentResult> documents = new ArrayList<>();
        Map<String, Integer> categoryCount = new HashMap<>();

        try (FileInputStream fis = new FileInputStream(excelFile);
             Workbook workbook = WorkbookFactory.create(fis)) {

            Sheet sheet = workbook.getSheetAt(0);
            System.out.println("Excel sheet: " + sheet.getSheetName() + ", filas físicas: " + sheet.getPhysicalNumberOfRows());
            if (sheet.getPhysicalNumberOfRows() <= 1) {
                return documents;
            }

            Row headerRow = sheet.getRow(0);
            if (headerRow == null) return documents;

            int speakerCol = -1;
            int textCol = -1;
            int idCol = -1;
            int dateCol = -1;

            for (Cell cell : headerRow) {
                String val = getCellValueAsString(cell).trim().toLowerCase();
                if (val.contains("speaker") || val.contains("author") || val.contains("autor") || val.contains("category")) {
                    speakerCol = cell.getColumnIndex();
                } else if (val.contains("text-original") || val.equals("text_clean") || val.equals("text") || val.contains("content")) {
                    if (textCol == -1 || val.contains("text-original")) {
                        textCol = cell.getColumnIndex();
                    }
                } else if (val.equals("id")) {
                    idCol = cell.getColumnIndex();
                } else if (val.contains("date") || val.contains("fecha")) {
                    dateCol = cell.getColumnIndex();
                }
            }

            if (textCol == -1) textCol = 6;
            if (speakerCol == -1) speakerCol = 2;

            System.out.println("Columnas detectadas -> Speaker: " + speakerCol + ", Text: " + textCol + ", ID: " + idCol + ", Date: " + dateCol);

            int rowCount = sheet.getLastRowNum();
            for (int r = 1; r <= rowCount; r++) {
                Row row = sheet.getRow(r);
                if (row == null) continue;

                String speaker = getCellValueAsString(row.getCell(speakerCol)).trim();
                String text = getCellValueAsString(row.getCell(textCol)).trim();
                String id = idCol != -1 ? getCellValueAsString(row.getCell(idCol)).trim() : String.valueOf(r);
                String date = dateCol != -1 ? getCellValueAsString(row.getCell(dateCol)).trim() : "";

                if (text.isEmpty() || text.length() < 10) continue;
                if (speaker.isEmpty()) speaker = "Desconocido";

                if (limitPerCategory > 0) {
                    int current = categoryCount.getOrDefault(speaker, 0);
                    if (current >= limitPerCategory) continue;
                    categoryCount.put(speaker, current + 1);
                }

                String docName = "doc_" + id + (date.isEmpty() ? "" : "_" + date.replaceAll("[^a-zA-Z0-9]", "_")) + ".txt";
                documents.add(new DocumentResult(speaker, docName, text));
            }

        } catch (IOException e) {
            System.err.println("Error leyendo archivo Excel: " + excelFile.getAbsolutePath() + " - " + e.getMessage());
        }

        return documents;
    }

    private static String getCellValueAsString(Cell cell) {
        if (cell == null) return "";
        CellType type = cell.getCellType();
        if (type == CellType.STRING) {
            return cell.getStringCellValue();
        } else if (type == CellType.NUMERIC) {
            double val = cell.getNumericCellValue();
            if (val == (long) val) {
                return String.valueOf((long) val);
            }
            return String.valueOf(val);
        } else if (type == CellType.BOOLEAN) {
            return String.valueOf(cell.getBooleanCellValue());
        } else if (type == CellType.FORMULA) {
            try {
                return cell.getStringCellValue();
            } catch (Exception e) {
                return String.valueOf(cell.getNumericCellValue());
            }
        }
        return "";
    }
}
