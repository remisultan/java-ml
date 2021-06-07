package org.rsultan.core.tree.impurity;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

import org.nd4j.linalg.api.ndarray.INDArray;

public class RmseService extends AbstractImpurityService {

  @Override
  public INDArray compute(INDArray Y) {
    var mse = pow(Y.sub(Y.mean()), 2).reshape(1, Y.length()).sum(true, 1).div(Y.columns());
    return sqrt(mse);
  }
}
