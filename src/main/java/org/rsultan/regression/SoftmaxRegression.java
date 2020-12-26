package org.rsultan.regression;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;

import static java.util.stream.Collectors.toList;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.log;


public class SoftmaxRegression extends GradientDescentRegression {

    public SoftmaxRegression(int numbersOfIterations, double alpha) {
        super(numbersOfIterations, alpha);
    }

    @Override
    public SoftmaxRegression setResponseVariableName(String name) {
        super.setResponseVariableName(name);
        return this;
    }

    @Override
    public SoftmaxRegression setPredictionColumnName(String name) {
        super.setPredictionColumnName(name);
        return this;
    }

    @Override
    public SoftmaxRegression setPredictorNames(String... names) {
        super.setPredictorNames(names);
        return this;
    }

    @Override
    public SoftmaxRegression train(Dataframe dataframe) {
        super.train(dataframe);
        return this;
    }

    protected INDArray computeNullHypothesis(INDArray X, INDArray W) {
        return computeSoftmax(X.mmul(W));
    }

    private INDArray computeSoftmax(INDArray z) {
        var exp = exp(z.sub(Nd4j.max(z)));
        return exp.div(exp.sum(true, 1));
    }

    protected INDArray computeGradient(
            INDArray X,
            INDArray Xt,
            INDArray W,
            INDArray labels) {
        var prediction = computeNullHypothesis(X, W);
        return Xt.div(X.rows()).mmul(prediction.sub(labels));
    }

    protected double computeLoss(INDArray predictions, INDArray Y) {
        var logLikelihood = log(predictions).mul(Y).sum(true, 1).neg();
        return logLikelihood.mean().getDouble(0, 0);
    }
}
