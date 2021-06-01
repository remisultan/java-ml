package org.rsultan.core.tree.impurity;

import java.io.Serializable;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface ImpurityService extends Serializable {

  INDArray compute(INDArray probabilities);

  INDArray getClassCount(INDArray labels);

}
