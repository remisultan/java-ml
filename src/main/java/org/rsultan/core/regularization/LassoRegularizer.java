package org.rsultan.core.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;

public class LassoRegularizer extends AbstractRegularizer {

    public LassoRegularizer(INDArray W, double lambda) {
        super(W, lambda);
    }

    @Override
    public double regularize() {
        return abs(W).sum().mul(lambda).getDouble(0, 0);
    }

    @Override
    public INDArray gradientRegularize() {
        return W.div(abs(W)).mul(lambda);
    }
}
