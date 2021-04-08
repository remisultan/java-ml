package org.rsultan.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static java.util.stream.IntStream.range;

public class Matrices {

    public static INDArray vectorAverage(INDArray m) {
        if(m.columns() > 1){
            throw new IllegalArgumentException("Vector must have 1 column");
        }
        double average = m.sumNumber().doubleValue() / (double) m.rows();
        double[] data = range(0, m.rows()).mapToDouble(num -> average).toArray();
        return Nd4j.create(data, m.shape());
    }

    public static INDArray diagonal(INDArray m) {
        if (m.isSquare()) {
            var diagonal = range(0, m.rows()).mapToDouble(integer -> m.getDouble(integer, integer)).toArray();
            return Nd4j.create(diagonal, m.rows(), 1);
        }
        throw new IllegalArgumentException("Matrix is not a square matrix, shape: " + Arrays
                .toString(m.shape()));
    }
}
