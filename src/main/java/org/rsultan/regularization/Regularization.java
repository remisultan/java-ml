package org.rsultan.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public enum Regularization {
    RIDGE, LASSO, NONE;

    public Regularizer getRegularizer(INDArray W, double lambda) {
        return switch (this) {
            case RIDGE -> new RidgeRegularizer(W, lambda);
            case LASSO -> new LassoRegularizer(W, lambda);
            default -> new AbstractRegularizer(W, lambda) {
                @Override
                public double regularize() {
                    return 0;
                }

                @Override
                public INDArray gradientRegularize() {
                    return Nd4j.zeros(W.shape());
                }
            };
        };
    }
}
