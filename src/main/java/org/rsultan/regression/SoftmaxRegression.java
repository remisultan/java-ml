package org.rsultan.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.function.Function;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.log;


public class SoftmaxRegression extends AbstractLogisticRegression {

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
    public INDArray computeNullHypothesis(INDArray X, INDArray W) {
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

    @Override
    public double computeLoss(INDArray predictions) {
        var logLikelihood = log(predictions).mul(YoneHot).sum(true, 1).neg();
        return logLikelihood.mean().getDouble(0, 0);
    }

    @Override
    protected  Function<String, String> formatPredictedLabel() {
        return String::valueOf;
    }
}
