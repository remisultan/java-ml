package org.rsultan.core.clustering.ensemble.domain;

import static java.util.Objects.nonNull;

import java.io.Serializable;
import org.nd4j.linalg.api.ndarray.INDArray;

public record IsolationNode(
    int feature,
    double featureThreshold,
    INDArray data,
    IsolationNode left,
    IsolationNode right
) implements Serializable {

  public IsolationNode(INDArray data) {
    this(-1, -1, data, null, null);
  }
  public IsolationNode(
      int feature,
      double featureThreshold,
      IsolationNode left,
      IsolationNode right) {
    this(feature, featureThreshold, null, left, right);
  }

  public boolean isLeaf() {
    return nonNull(data);
  }
}