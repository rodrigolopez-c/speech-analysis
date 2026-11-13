package com.nlp.pipeline.vectorization;

import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class PCAReducer {

    public static double[][] applyPCA(List<double[]> data, int numComponents) {

        int rows = data.size();
        if (rows == 0) return new double[0][0];
        int cols = data.get(0).length;

        double[][] matrixData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            matrixData[i] = data.get(i).clone();
        }

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

        RealMatrix centered = new Array2DRowRealMatrix(matrixData);

        if (rows < cols) {
            // Matriz Gram X * X^T de tamaño (rows x rows) -> 694 x 694
            RealMatrix gram = centered.multiply(centered.transpose());
            EigenDecomposition ed = new EigenDecomposition(gram);

            double[][] result = new double[rows][numComponents];
            for (int k = 0; k < numComponents; k++) {
                double[] vec = ed.getEigenvector(k).toArray();
                double val = ed.getRealEigenvalue(k);
                double scale = val > 0 ? Math.sqrt(val) : 1.0;
                for (int i = 0; i < rows; i++) {
                    result[i][k] = vec[i] * scale;
                }
            }
            return result;
        } else {
            SingularValueDecomposition svd = new SingularValueDecomposition(centered);
            RealMatrix V = svd.getV();
            RealMatrix principalComponents = V.getSubMatrix(0, cols - 1, 0, numComponents - 1);
            RealMatrix reduced = centered.multiply(principalComponents);
            return reduced.getData();
        }
    }
}
