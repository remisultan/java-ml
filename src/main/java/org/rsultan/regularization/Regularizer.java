package org.rsultan.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Regularizer {

    double regularize();

    INDArray gradientRegularize();
}
