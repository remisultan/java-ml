package org.rsultan.core.tree.impurity;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;

import org.nd4j.linalg.api.ndarray.INDArray;

public class GiniService extends AbstractImpurityService {

  @Override
  public INDArray compute(INDArray classCount) {
    var probabilities = computeProbabilities(classCount.reshape(1, classCount.length()));
    return pow(probabilities, 2).sum(true, 1).neg().add(1);
  }
}
