package org.rsultan.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;

public class RidgeRegularizer extends AbstractRegularizer {

    public RidgeRegularizer(INDArray W, double lambda) {
        super(W, lambda);
    }

    @Override
    public double regularize() {
        return W.transpose().mmul(W).mul(lambda / 2.0D).getDouble(0, 0);
    }

    @Override
    public INDArray gradientRegularize() {
        return W.mul(lambda);
    }
}
