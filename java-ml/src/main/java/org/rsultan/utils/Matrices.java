package org.rsultan.utils;

import static java.util.stream.IntStream.range;

import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Matrices {

  public static INDArray vectorAverage(INDArray m) {
    if (m.columns() > 1) {
      throw new IllegalArgumentException("Vector must have 1 column");
    }
    double average = m.sumNumber().doubleValue() / (double) m.rows();
    double[] data = range(0, m.rows()).mapToDouble(num -> average).toArray();
    return Nd4j.create(data, m.shape());
  }

  public static INDArray diagonal(INDArray m) {
    if (m.isSquare()) {
      var diagonal = range(0, m.rows()).mapToDouble(integer -> m.getDouble(integer, integer))
          .toArray();
      return Nd4j.create(diagonal, m.rows(), 1);
    }
    throw new IllegalArgumentException("Matrix is not square, shape: " + Arrays
        .toString(m.shape()));
  }

  public static INDArray covariance(INDArray m) {
    var mDemeaned = m.sub(m.mean(0));
    return range(0, mDemeaned.rows())
        .mapToObj(mDemeaned::getRow)
        .map(vector -> vector.reshape(m.columns(), 1))
        .map(row -> row.mmul(row.transpose()))
        .reduce(INDArray::add)
        .orElse(Nd4j.ones(m.columns(), m.columns()))
        .divi(m.rows());
  }
}
