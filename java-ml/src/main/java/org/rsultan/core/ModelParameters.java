package org.rsultan.core;


import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.io.Serializable;
import java.security.SecureRandom;
import java.util.Collections;
import java.util.stream.Stream;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class ModelParameters<T> implements Serializable {

  private static final SecureRandom RND = new SecureRandom();
  protected String responseVariableName = "y";
  protected String predictionColumnName = "predictions";
  protected String[] predictorNames = {};

  protected boolean shuffle = false;

  public T setResponseVariableName(String responseVariableName) {
    this.responseVariableName = responseVariableName;
    return (T) this;
  }

  public T setPredictionColumnName(String name) {
    this.predictionColumnName = name;
    return (T) this;
  }

  public T setPredictorNames(String... names) {
    this.predictorNames = Stream.of(names).distinct().toArray(String[]::new);
    return (T) this;
  }

  public T setShuffle(boolean shuffle) {
    this.shuffle = shuffle;
    return (T) this;
  }

  protected void shuffle(INDArray... ndArrays) {
    if (shuffle && ndArrays != null && ndArrays.length > 0) {
      var indices = range(0, ndArrays[0].rows()).boxed().collect(toList());
      final int[] rows = indices.stream().mapToInt(i -> i).toArray();
      Collections.shuffle(indices);
      for (INDArray indArray : ndArrays) {
        final INDArray rows1 = indArray.getRows(rows);
        indArray.muli(0).addi(rows1);
      }
    }
  }
}
