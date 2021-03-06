package org.rsultan.core;

import java.io.Serializable;
import java.util.stream.Stream;

public abstract class ModelParameters<T> implements Serializable {

  protected String responseVariableName = "y";
  protected String predictionColumnName = "predictions";
  protected String[] predictorNames = {};

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
}
