package org.rsultan.core.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Regularizer {

    double regularize();

    INDArray gradientRegularize();
}
