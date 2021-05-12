package org.rsultan.core.tree.impurity;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

import org.nd4j.linalg.api.ndarray.INDArray;

public class RmseService extends AbstractImpurityService {

  public RmseService(int totalLabels) {
    super(totalLabels);
  }

  @Override
  public INDArray compute(INDArray Y) {
    var probabilities = computeProbabilities(Y);
    var mse = pow(Y.sub(Y.mean()), 2).sum(true, 1).div(Y.columns());
    return sqrt(probabilities.mul(mse).sum());
  }
}
