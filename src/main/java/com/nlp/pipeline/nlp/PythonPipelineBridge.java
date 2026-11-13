package com.nlp.pipeline.nlp;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Puente para ejecutar los módulos avanzados de Python (Transformers, DeepMoji, Scikit-Learn y BiLSTM/WordNet)
 * directamente desde la ejecución del pipeline en Java.
 */
public class PythonPipelineBridge {

    private final String pythonExecutable;

    public PythonPipelineBridge() {
        this.pythonExecutable = detectPythonExecutable();
    }

    private String detectPythonExecutable() {
        // En Windows verificar python o py
        String[] candidates = {"python", "py", "python3"};
        for (String cmd : candidates) {
            try {
                Process p = new ProcessBuilder(cmd, "--version").start();
                if (p.waitFor() == 0) {
                    return cmd;
                }
            } catch (Exception ignored) {
            }
        }
        return "python";
    }

    public boolean executePythonModule(String scriptPath, String... args) {
        File script = new File(scriptPath);
        if (!script.exists()) {
            System.err.println("Script de Python no encontrado: " + script.getAbsolutePath());
            return false;
        }

        List<String> command = new ArrayList<>();
        command.add(pythonExecutable);
        command.add(script.getAbsolutePath());
        for (String arg : args) {
            command.add(arg);
        }

        System.out.println("\n[Python Bridge] Ejecutando: " + String.join(" ", command));

        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("  [Python] " + line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("[Python Bridge] Módulo finalizado exitosamente.");
                return true;
            } else {
                System.err.println("[Python Bridge] Error: proceso finalizó con código " + exitCode);
                return false;
            }
        } catch (Exception e) {
            System.err.println("[Python Bridge] Excepción ejecutando módulo de Python: " + e.getMessage());
            return false;
        }
    }
}
