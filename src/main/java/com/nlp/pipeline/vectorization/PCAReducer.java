package com.nlp.pipeline.vectorization;

import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class PCAReducer {

    public static double[][] applyPCA(List<double[]> data, int numComponents) {

        int rows = data.size();
        int cols = data.get(0).length;

        double[][] matrixData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            matrixData[i] = data.get(i);
        }

        RealMatrix matrix = MatrixUtils.createRealMatrix(matrixData);

        // 🔹 Centrar datos
        for (int j = 0; j < cols; j++) {
            double mean = 0;
            for (int i = 0; i < rows; i++) {
                mean += matrixData[i][j];
            }
            mean /= rows;

            for (int i = 0; i < rows; i++) {
                matrixData[i][j] -= mean;
            }
        }

        RealMatrix centered = MatrixUtils.createRealMatrix(matrixData);

        // 🔥 SVD (estable)
        SingularValueDecomposition svd = new SingularValueDecomposition(centered);

        RealMatrix V = svd.getV();

        // Tomar las primeras componentes
        RealMatrix principalComponents =
                V.getSubMatrix(0, cols - 1, 0, numComponents - 1);

        // Proyección
        RealMatrix reduced = centered.multiply(principalComponents);

        return reduced.getData();
    }
}