package org.rsultan.core.tree.impurity;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ImpurityService {

  INDArray compute(INDArray probabilities);

  INDArray getClassCount(INDArray labels);

}
