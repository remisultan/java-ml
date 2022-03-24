package org.rsultan.core.ensemble.isolationforest.domain;

import static java.util.Objects.nonNull;

import java.io.Serializable;
import org.nd4j.linalg.api.ndarray.INDArray;

public record IsolationNode<T>(
    T nodeData,
    INDArray data,
    IsolationNode<T> left,
    IsolationNode<T> right
) implements Serializable {

  public IsolationNode(INDArray data) {
    this(null, data, null, null);
  }
  public IsolationNode(
      T nodeData,
      IsolationNode<T> left,
      IsolationNode<T> right) {
    this(nodeData, null, left, right);
  }

  public boolean isLeaf() {
    return nonNull(data);
  }
}