package org.rsultan.core.tree.impurity;

import static org.nd4j.linalg.indexing.BooleanIndexing.replaceWhere;
import static org.nd4j.linalg.indexing.conditions.Conditions.isInfinite;
import static org.nd4j.linalg.ops.transforms.Transforms.log;

import org.nd4j.linalg.api.ndarray.INDArray;

public class EntropyService extends AbstractImpurityService {

  public EntropyService(int totalLabels) {
    super(totalLabels);
  }

  @Override
  public INDArray compute(INDArray classCount) {
    var probabilities = computeProbabilities(classCount);
    var logProb = log(probabilities, 2, true).neg();
    replaceWhere(logProb, 0.0, isInfinite());
    return probabilities.mul(logProb).sum(true, 1);
  }

}
