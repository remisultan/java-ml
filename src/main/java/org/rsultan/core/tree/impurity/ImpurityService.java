package org.rsultan.core.tree.impurity;

import java.io.Serializable;
import java.util.Map;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface ImpurityService extends Serializable {

  INDArray compute(INDArray probabilities);

  Map<Double, Long> getClassCount(INDArray labels);

}
