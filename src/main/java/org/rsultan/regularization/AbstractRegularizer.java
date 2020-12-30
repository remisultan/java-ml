package org.rsultan.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class AbstractRegularizer implements Regularizer {

    protected final INDArray W;
    protected final double lambda;

    protected AbstractRegularizer(INDArray W, double lambda) {
        this.W = W;
        this.lambda = lambda;
    }
}
